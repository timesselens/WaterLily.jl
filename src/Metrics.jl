using StaticArrays

# utilities
# Each f(i) gets a call-site @inline, which ntuple's internal calls cannot take
Base.@propagate_inbounds @inline _tuple_inl(f,::Val{0}) = ()
Base.@propagate_inbounds @inline _tuple_inl(f,::Val{N}) where N = (_tuple_inl(f,Val(N-1))...,@inline f(N))
Base.@propagate_inbounds @inline fSV(f,n) = SVector(_tuple_inl(f,Val(n)))
Base.@propagate_inbounds @inline @fastmath fsum(f,n) = sum(ntuple(f,n))
norm2(x) = √(x'*x)
"""
    shiftDir(d,D,i)

Shift the index of direction `d` to `i` steps away in dimension space of `D`.
So `shiftDir(1,3,2) = 3`, `shiftDir(1,4,-1) = 4`
"""
shiftDir(d,D,i) = mod(d+i-1,D)+1
Base.@propagate_inbounds @fastmath function permute(f,i)
    j,k = shiftDir(i,3,1), shiftDir(i,3,2)
    @inline f(j,k)-f(k,j)
end
×(a,b) = fSV(i->permute((j,k)->a[j]*b[k],i),3)
@fastmath @inline function dot(a,b)
    init=zero(eltype(a))
    @inbounds for ij in eachindex(a)
     init += a[ij] * b[ij]
    end
    return init
end

"""
    ke(I::CartesianIndex,u,U=0)

Compute ``½∥𝐮-𝐔∥²`` at center of cell `I` where `U` can be used
to subtract a background flow (by default, `U=0`).
"""
ke(I::CartesianIndex{m},u,U=fSV(zero,m)) where m = fsum(m) do i
    abs2(@inbounds(u[I,i]+u[I+δ(i,I),i]-2U[i]))/8
end
"""
    ∂(i,j,I,u)

Compute ``∂uᵢ/∂xⱼ`` at center of cell `I`. Cross terms are computed
less accurately than inline terms because of the staggered grid.
"""
@fastmath @inline ∂(i,j,I,u) = (i==j ? ∂(i,I,u) :
        @inbounds(u[I+δ(j,I),i]+u[I+δ(j,I)+δ(i,I),i]
                 -u[I-δ(j,I),i]-u[I-δ(j,I)+δ(i,I),i])/4)

using LinearAlgebra: eigvals, Hermitian
"""
    λ₂(I::CartesianIndex{3},u)

λ₂ is a deformation tensor metric to identify vortex cores.
See [https://en.wikipedia.org/wiki/Lambda2_method](https://en.wikipedia.org/wiki/Lambda2_method) and
Jeong, J., & Hussain, F., doi:[10.1017/S0022112095000462](https://doi.org/10.1017/S0022112095000462)
"""
function λ₂(I::CartesianIndex{3},u)
    J = @SMatrix [∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S,Ω = (J+J')/2,(J-J')/2
    eigvals(Hermitian(S^2+Ω^2))[2]
end

"""
    curl(i,I,u)

Compute component `i` of ``𝛁×𝐮`` at the __edge__ of cell `I`.
For example `curl(3,CartesianIndex(2,2,2),u)` will compute
`ω₃(x=1.5,y=1.5,z=2)` as this edge produces the highest
accuracy for this mix of cross derivatives on a staggered grid.
"""
curl(i,I,u) = permute((j,k)->∂(j,CI(I,k),u), i)
"""
    ω(I::CartesianIndex{3},u)

Compute 3-vector ``𝛚=𝛁×𝐮`` at the center of cell `I`.
"""
@inline ω(I::CartesianIndex{3},u) = fSV(i->permute((j,k)->∂(k,j,I,u),i),3)
"""
    ω(I::CartesianIndex{2},u)

Compute the scalar out-of-plane vorticity ``ω₃=∂u₂/∂x₁-∂u₁/∂x₂`` at the center of cell `I`.
"""
@inline ω(I::CartesianIndex{2},u) = permute((j,k)->∂(k,j,I,u),3)
"""
    ω_mag(I::CartesianIndex{3},u)

Compute ``∥𝛚∥`` at the center of cell `I`.
"""
@inline ω_mag(I::CartesianIndex{3},u) = norm2(ω(I,u))
"""
    ω_θ(I::CartesianIndex{3},z,center,u)

Compute ``𝛚⋅𝛉`` at the center of cell `I` where ``𝛉`` is the azimuth
direction around vector `z` passing through `center`.
`z` and `center` are converted to the element type of `u`.
"""
@inline function ω_θ(I::CartesianIndex{3},z,center,u)
    T = eltype(u)
    θ = SVector{3,T}(z) × (loc(0,I,T)-SVector{3,T}(center))
    n = norm2(θ)
    n<=eps(n) ? zero(n) : θ'*ω(I,u) / n
end

"""
    helicity(I,u,ω)

Compute the helicity density at collocated cell `I` from dot product of velocity `u` and vorticity `ω`.
`u` and `ω` should all be average to the collocated cell in each directional operation.
"""
function helicity(I::CartesianIndex{3},u::AbstractArray{T},ω) where T
    s = zero(T)
    for d∈1:3
        dir1,dir2 = shiftDir.(d,3,1:2)
        umid = u[I,d]+u[I+δ(d,I),d]
        for id1∈0:1,id2∈0:1
            s+=umid*ω[I+id1*δ(dir1,I)+id2*δ(dir2,I),d]
        end
    end
    s/8
end

"""
    nds(body,x,t)

BDIM-masked surface normal at `x`, the surface element `n ds` that the cell centred there holds: the jump of the
fluid fraction `μ₀(d)` across the cell,

    nds = Σᵢ eᵢ [μ₀(d(x+eᵢ/2)) - μ₀(d(x-eᵢ/2))].

Summed over the cells it telescopes: a uniform pressure or stress gives exactly zero force. Outside `measure!`'s band,
|d| ≥ 3, the fraction is 0 or 1 on every face, so only the cells inside the band are measured.
"""
@inline function nds(body,x::AbstractVector{T},t) where T
    D = length(x); d(x) = T(sdf(body,x,t,fastd²=T(9)))
    e(i) = SVector(ntuple(j->T(j==i)/2,D))
    abs(d(x)) < 3 ? fSV(i->μ₀(d(x+e(i)),one(T))-μ₀(d(x-e(i)),one(T)),D) : zero(SVector{D,T})
end

"""
    Measured(sim)

The body as `measure!` left it in `sim.flow`: the fluid fraction `μ₀` of every cell face, for the kernel width `ϵ`.
The force and moment integrals read these fractions instead of measuring the body again. For the Poisson solve,
`BC!` sets the fraction of the domain's boundary faces to zero. Near the body those faces are measured again; further
out they take the value of the next face in, which there is 0 or 1.
"""
struct Measured{A,P,B,T} <: AbstractBody
    μ₀ :: A
    perdir :: P
    body :: B
    ϵ :: T
end
Measured(sim) = Measured(sim.flow.μ₀,sim.flow.perdir,sim.body,eltype(sim.flow.p)(sim.ϵ))
measure(m::Measured,x,t;kwargs...) = measure(m.body,x,t;kwargs...)
using Adapt: @adapt_structure
@adapt_structure Measured

"""
    nds(body,I,x,t)

`nds(body,x,t)` of the cell `I` centred at `x`. For `Measured`, the jump of the stored fractions across the cell;
cells outside `measure!`'s band, |d| ≥ 2+ϵ, are skipped, as there the fractions are 0 or 1.
"""
@inline nds(body,I,x,t) = nds(body,x,t)
@inline function nds(m::Measured,I::CartesianIndex{D},x::AbstractVector{T},t) where {D,T}
    e(i) = SVector(ntuple(j->T(j==i)/2,D))
    near(m.body,x,t,m.ϵ) ? fSV(i->μ₀(m,I+δ(i,I),i,x+e(i),t)-μ₀(m,I,i,x-e(i),t),D) : zero(SVector{D,T})
end
@inline μ₀(m::Measured,J,i,y,t) = onboundary(m,J,i) ? remeasured(m,J,i,y,t) : @inbounds m.μ₀[J,i]
@inline onboundary(m,J,i) = i ∉ m.perdir && J[i] ∈ (2,size(m.μ₀,i))
@inline remeasured(m,J,i,y,t) = near(m.body,y,t,m.ϵ) ? μ₀(oftype(m.ϵ,sdf(m.body,y,t,fastd²=(2+m.ϵ)^2)),m.ϵ) : @inbounds m.μ₀[inward(m,J,i),i]
@inline inward(m,J,i) = CIj(i,J,clamp(J[i],3,size(m.μ₀,i)-1))
@inline near(body,x,t,ϵ) = abs(sdf(body,x,t,fastd²=(2+ϵ)^2)) < 2+ϵ

"""
    sumdims(df)

Sum the workspace `df` over the domain in its own element type, returning one value per component.
"""
sumdims(df) = sum(df,dims=ntuple(i->i,ndims(df)-1))[:] |> Array

"""
    pressure_force(sim)

Compute the pressure force on an immersed body.
"""
pressure_force(sim) = pressure_force(sim.flow,Measured(sim))
pressure_force(flow,body) = pressure_force(flow.p,flow.f,body,time(flow))
function pressure_force(p,df,body,t=0)
    df .= zero(eltype(p))
    @loop df[I,:] .= p[I]*nds(body,I,loc(0,I,eltype(p)),t) over I ∈ bbox(body,inside(p))
    sumdims(df)
end

"""
    S(I::CartesianIndex,u)

Rate-of-strain tensor.
"""
@inline S(I::CartesianIndex{D},u) where D = SMatrix{D,D}((∂(i,j,I,u)+∂(j,i,I,u))/2 for i ∈ 1:D, j ∈ 1:D)
@inline Snds(I,u,body,t) = (n = nds(body,I,loc(0,I,eltype(u)),t); iszero(n) ? n : S(I,u)*n)

"""
    viscous_force(sim)

Compute the viscous force on an immersed body.
"""
viscous_force(sim) = viscous_force(sim.flow,Measured(sim))
viscous_force(flow,body) = viscous_force(flow.u,flow.ν,flow.f,body,time(flow))
function viscous_force(u,ν,df,body,t=0)
    df .= zero(eltype(u))
    @loop df[I,:] .= -2ν*Snds(I,u,body,t) over I ∈ bbox(body,inside_u(u))
    sumdims(df)
end

"""
    τ(I,p,u,ν)

Total stress tensor (pressure and viscous) acting on the body surface, so that `τ⋅nds` is the force density.
"""
@inline τ(I::CartesianIndex{D},p,u,ν) where D = (s = S(I,u); p[I]*one(s)-2ν*s)
@inline τnds(I,p,u,ν,body,t) = (n = nds(body,I,loc(0,I,eltype(p)),t); iszero(n) ? n : τ(I,p,u,ν)*n)

"""
    total_force(sim)

Compute the total force on an immersed body.
"""
total_force(sim) = total_force(sim.flow,Measured(sim))
total_force(flow,body) = total_force(flow.p,flow.u,flow.ν,flow.f,body,time(flow))
function total_force(p,u,ν,df,body,t=0)
    df .= zero(eltype(p))
    @loop df[I,:] .= τnds(I,p,u,ν,body,t) over I ∈ bbox(body,inside(p))
    sumdims(df)
end

using LinearAlgebra: cross
"""
    pressure_moment(x₀,sim)

Computes the pressure moment on an immersed body relative to point x₀.
`x₀` is converted to the element type of the pressure field.
"""
pressure_moment(x₀,sim) = pressure_moment(x₀,sim.flow,Measured(sim))
pressure_moment(x₀,flow,body) = pressure_moment(x₀,flow.p,flow.f,body,time(flow))
function pressure_moment(x₀,p,df,body,t=0)
    df .= zero(eltype(p)); x₀ = eltype(p).(x₀)
    @loop df[I,:] .= p[I]*cross(loc(0,I,eltype(p))-x₀,nds(body,I,loc(0,I,eltype(p)),t)) over I ∈ bbox(body,inside(p))
    sumdims(df)
end

"""
    viscous_moment(x₀,sim)

Computes the viscous moment on an immersed body relative to point x₀.
`x₀` is converted to the element type of the velocity field.
"""
viscous_moment(x₀,sim) = viscous_moment(x₀,sim.flow,Measured(sim))
viscous_moment(x₀,flow,body) = viscous_moment(x₀,flow.u,flow.ν,flow.f,body,time(flow))
function viscous_moment(x₀,u,ν,df,body,t=0)
    df .= zero(eltype(u)); x₀ = eltype(u).(x₀)
    @loop df[I,:] .= -2ν*cross(loc(0,I,eltype(u))-x₀,Snds(I,u,body,t)) over I ∈ bbox(body,inside_u(u))
    sumdims(df)
end

"""
    total_moment(x₀,sim)

Computes the total (pressure + viscous) moment on an immersed body relative to point x₀.
"""
total_moment(x₀,sim) = total_moment(x₀,sim.flow,Measured(sim))
total_moment(x₀,flow,body) = total_moment(x₀,flow.p,flow.u,flow.ν,flow.f,body,time(flow))
function total_moment(x₀,p,u,ν,df,body,t=0)
    df .= zero(eltype(p)); x₀ = eltype(p).(x₀)
    @loop df[I,:] .= cross(loc(0,I,eltype(p))-x₀,τnds(I,p,u,ν,body,t)) over I ∈ bbox(body,inside(p))
    sumdims(df)
end

"""
    total_force_and_moment(x₀,sim)

Computes the total force and the total moment relative to point x₀ in a single pass over the body.
The moment is accumulated in `flow.u⁰`, which is free between time steps.
"""
total_force_and_moment(x₀,sim) = total_force_and_moment(x₀,sim.flow,Measured(sim))
total_force_and_moment(x₀,flow,body) = total_force_and_moment(x₀,flow.p,flow.u,flow.ν,flow.f,flow.u⁰,body,time(flow))
function total_force_and_moment(x₀,p,u,ν,df,dm,body,t=0)
    df .= zero(eltype(p)); dm .= zero(eltype(p)); x₀ = eltype(p).(x₀)
    @loop force_moment!(df,dm,I,p,u,ν,x₀,body,t) over I ∈ bbox(body,inside(p))
    sumdims(df), sumdims(dm)
end
@inline function force_moment!(df,dm,I,p,u,ν,x₀,body,t)
    F = τnds(I,p,u,ν,body,t)
    df[I,:] .= F; dm[I,:] .= cross(loc(0,I,eltype(p))-x₀,F)
end

"""
    MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf<:AbstractArray{T}}

Holds temporal averages of pressure, velocity, and squared-velocity tensor.
"""
struct MeanFlow{T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Mf}
    P :: Sf # pressure scalar field
    U :: Vf # velocity vector field
    UU :: Mf # squared-velocity tensor, u⊗u
    t :: Vector{T} # time steps vector
    uu_stats :: Bool # flag to compute UU on-the-fly temporal averages
    function MeanFlow(flow::AbstractFlow{D,T}; t_init=time(flow), uu_stats=false) where {D,T}
        mem = typeof(flow.u).name.wrapper
        P = zeros(T, size(flow.p)) |> mem
        U = zeros(T, size(flow.u)) |> mem
        UU = uu_stats ? zeros(T, size(flow.p)..., D, D) |> mem : nothing
        new{T,typeof(P),typeof(U),typeof(UU)}(P,U,UU,T[t_init],uu_stats)
    end
    function MeanFlow(N::NTuple{D}; mem=Array, T=Float32, t_init=0, uu_stats=false) where {D}
        Ng = N .+ 2
        P = zeros(T, Ng) |> mem
        U = zeros(T, Ng..., D) |> mem
        UU = uu_stats ? zeros(T, Ng..., D, D) |> mem : nothing
        new{T,typeof(P),typeof(U),typeof(UU)}(P,U,UU,T[t_init],uu_stats)
    end
end

time(meanflow::MeanFlow) = meanflow.t[end]-meanflow.t[1]

function reset!(meanflow::MeanFlow; t_init=0.0)
    fill!(meanflow.P, 0); fill!(meanflow.U, 0)
    !isnothing(meanflow.UU) && fill!(meanflow.UU, 0)
    deleteat!(meanflow.t, collect(1:length(meanflow.t)))
    push!(meanflow.t, t_init)
end

function update!(meanflow::MeanFlow, flow::AbstractFlow)
    dt = time(flow) - meanflow.t[end]
    ε = dt / (dt + time(meanflow) + eps(eltype(flow.p)))
    length(meanflow.t) == 1 && (ε = 1) # if it's the first update, we just take the instantaneous flow field
    @loop meanflow.P[I] = ε * flow.p[I] + (1 - ε) * meanflow.P[I] over I in CartesianIndices(flow.p)
    @loop meanflow.U[Ii] = ε * flow.u[Ii] + (1 - ε) * meanflow.U[Ii] over Ii in CartesianIndices(flow.u)
    if meanflow.uu_stats
        for i in 1:ndims(flow.p), j in 1:ndims(flow.p)
            @loop meanflow.UU[I,i,j] = ε * (flow.u[I,i] .* flow.u[I,j]) + (1 - ε) * meanflow.UU[I,i,j] over I in CartesianIndices(flow.p)
        end
    end
    push!(meanflow.t, meanflow.t[end] + dt)
end

uu!(τ,a::MeanFlow) = for i in 1:ndims(a.P), j in 1:ndims(a.P)
    @loop τ[I,i,j] = a.UU[I,i,j] - a.U[I,i] * a.U[I,j] over I in CartesianIndices(a.P)
end
function uu(a::MeanFlow)
    τ = zeros(eltype(a.UU), size(a.UU)...) |> typeof(a.UU).name.wrapper
    uu!(τ,a)
    return τ
end

function copy!(a::AbstractFlow, b::MeanFlow)
    a.u .= b.U
    a.p .= b.P
end