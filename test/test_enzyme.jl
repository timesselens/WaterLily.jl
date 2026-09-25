# Reverse-mode AD with Enzyme (WaterLilyEnzymeCoreExt). Opt-in: WATERLILY_ENZYME=true (see runtests.jl).
using Enzyme, ForwardDiff, LinearAlgebra, Random
using WaterLily: @loop, sum_view, max_all, sum_dims, update!, mult!

const rev = set_runtime_activity(Reverse)

# The float type a backend is tested in: the widest it supports, decided the way `sumtype` (src/Metrics.jl) decides:
# Float64 on the CPU, CUDA and AMDGPU, Float32 on Metal, which has no Float64
testfloat(mem) = WaterLily.supports_float64(WaterLily.get_backend(mem(zeros(Float32, 1)))) ? Float64 : Float32
# a tolerance per precision: `f32` in Float32, `f64` otherwise (Float64, and ForwardDiff's duals of Float64)
tol(T; f64, f32) = T == Float32 ? f32 : f64

# @loop kernels (launch! rule): a stencil, whose adjoint scatters, with an active scalar `s` and an
# item function that captures an active value θ through `f`
function stencil(a, b, s, θ)
    f = x -> sin(θ * x)
    @loop b[I] = s * f(a[I-δ(1,I)] - 2a[I] + a[I+δ(1,I)]) over I ∈ inside(b)
    sum_view(b, inside(b))
end
# the same, with an item function that also captures a type (a Julia object, not a bits value: CPU only)
function stencil_T(a, b, s, θ)
    T = eltype(a); f = x -> sin(θ * x) * one(T)
    @loop b[I] = s * f(a[I-δ(1,I)] - 2a[I] + a[I+δ(1,I)]) over I ∈ inside(b)
    sum_view(b, inside(b))
end
# an argument holding both an array and an active value: not supported by the launch! rule, with a clear error
struct ArrayAndScale{A,T}
    a::A
    s::T
end
scaled(p, I) = p.s * p.a[I]
function mixed(a, b, s)
    p = ArrayAndScale(a, s)
    @loop b[I] = scaled(p, I) over I ∈ inside(b)
    sum_view(b, inside(b))
end
# reductions: the maximum (one-hot at the argmax) and a sum over the leading dimensions (host vector)
reductions(a, w) = 2max_all(a) + sum(sum_dims(eltype(a), a, Val(2)) .* w)

# Poisson solve (implicit-function rule) on random coefficients, with the exact solve on the active cells
function poisson_case(rng, perdir, masked; N=16)
    Ng = (N+2, N+2)
    L = 0.5 .+ 0.5rand(rng, Ng..., 2)
    masked && (L[7:10, 7:9, 1] .= 0; L[7:9, 7:10, 2] .= 0)
    BC!(L, zeros(SVector{2,Float64}), false, perdir)
    p = MultiLevelPoisson(zeros(Ng), L, zeros(Ng); perdir)
    act = [I ∈ inside(p.x) && p.levels[1].iD[I] != 0 for I ∈ CartesianIndices(Ng)]
    z = randn(rng, Ng) .* act; z .-= act .* (sum(z) / count(act)) # compatible (zero-mean) source
    w = randn(rng, Ng) .* act; w .-= act .* (sum(w) / count(act))
    copyto!(p.z, z)
    return p, w, act
end
function poisson_J(p, w)
    update!(p); fill!(p.x, 0); solver!(p; tol=1e-12, itmx=800)
    s = zero(eltype(w))
    for I ∈ inside(p.x); s += w[I] * p.x[I]; end
    return s
end
function poisson_dense(p, act)  # A on the active cells (from mult!) with the nullspace fixed: A + eeᵀ/n
    update!(p)
    idx = findall(act); n = length(idx); e = zeros(size(p.x)); f = p.levels[1]
    A = zeros(n, n)
    for (c, J) ∈ enumerate(idx)
        fill!(e, 0); e[J] = 1
        A[:, c] = mult!(f, e)[idx]
    end
    return A + ones(n, n) / n, idx
end
# exact dJ/dL[I,i] of J = ⟨w, A⁺z⟩: A is linear in L, so ∂A/∂L[I,i] = (A(L+1) - A(L-1))/2 exactly, and
# dJ = -λᵀ(∂A)x with λ = A⁺w (finite differences of J are not accurate enough: |J| ≈ 50, dJ/dL ≈ 1e-3)
function poisson_dJdL(p, w, act, z, I, i)
    A, idx = poisson_dense(p, act)
    x = A \ (z[idx] .- sum(z[idx]) / length(idx)); λ = A \ w[idx]
    L₀ = p.L[I, i]
    p.L[I, i] = L₀ + 1; Ap, _ = poisson_dense(p, act)
    p.L[I, i] = L₀ - 1; Am, _ = poisson_dense(p, act); p.L[I, i] = L₀
    return -dot(λ, (Ap - Am) / 2 * x)
end

# a flat plate rotated by θ (as in the ForwardDiff tests); J = Σ|u|² after the steps
function plate(θ, L)
    s, c = sincos(θ); h = oftype(θ, L/2); C = SA[oftype(θ, L), oftype(θ, L)]
    AutoBody((ξ, _) -> √sum(abs2, ξ - SA[zero(h), clamp(ξ[1], -h, h)]) - 2, (x, _) -> SA[c -s; s c] * (x - C))
end
plate_sim(θ; L=16, mem=Array) = Simulation((2L, 2L), (1, 0), L; ν=L/100, body=plate(θ, L), T=typeof(θ), mem)
# the pressure solves are converged tightly (the gradients are as accurate as the solves), as far as the float type allows
solve_kw(sim) = (; tol = tol(eltype(sim.flow.p); f64=1e-10, f32=1e-5), itmx = 500)
function energy(sim)
    a = sim.flow
    @loop a.σ[I] = abs2(a.u[I,1]) + abs2(a.u[I,2]) over I ∈ inside(a.p)
    sum_view(a.σ, inside(a.p))
end
one_step(sim, θ) = (sim.body = plate(θ, sim.L); measure!(sim);
    sim_step!(sim; remeasure=false, poisson_kwargs=solve_kw(sim)); energy(sim))
three_steps(sim, θ) = (sim.body = plate(θ, sim.L); measure!(sim);
    sim_step!(sim, 1e9; max_steps=3, poisson_kwargs=solve_kw(sim)); energy(sim)) # through evolve!
running_cost(sim, θ) = (sim.body = plate(θ, sim.L); measure!(sim);
    evolve!(sim, 1e9; max_steps=3, init=zero(θ)) do s
        sim_step!(s; remeasure=false, poisson_kwargs=solve_kw(s))
        s.flow.Δt[end-1] * WaterLily.pressure_force(s)[1]
    end)
# the same with the total force (pressure and viscous) in the cost
running_force(sim, θ) = (sim.body = plate(θ, sim.L); measure!(sim);
    evolve!(sim, 1e9; max_steps=2, init=zero(θ)) do s
        sim_step!(s; remeasure=false, poisson_kwargs=solve_kw(s))
        s.flow.Δt[end-1] * WaterLily.total_force(s)[2]
    end)
# a cost that doesn't infer (`sim_time`: `Simulation`'s U and L are abstract): the evolve! rule stops with an error
unstable_cost(sim, θ) = (sim.body = plate(θ, sim.L); measure!(sim);
    oftype(θ, evolve!(sim, 1e9; max_steps=1, init=zero(θ)) do s
        sim_step!(s; remeasure=false)
        sim_time(s) * WaterLily.pressure_force(s)[1]
    end))
# a circle whose radius is a parameter inside the sdf, off the grid's symmetry: its pressure force depends, at grid
# level, on the pressure's constant, so this checks that the constant (pinned after every solve) gets no derivative
circle_r(r, L) = AutoBody((x, _) -> √sum(abs2, x - SA[oftype(r, 1.6L), oftype(r, 0.95L)]) - r*L)
circle_sim(r; L=16, mem=Array) = Simulation((4L, 2L), (1, 0), L; ν=L/100, body=circle_r(r, L), T=typeof(r), mem)
drag_step(sim, r) = (sim.body = circle_r(r, sim.L); measure!(sim);
    sim_step!(sim; remeasure=false, poisson_kwargs=solve_kw(sim)); WaterLily.pressure_force(sim)[1])
function enzyme_dθ(J, θ; kw...)
    sim = plate_sim(θ; kw...)
    autodiff(rev, J, Active, Duplicated(sim, Enzyme.make_zero(sim)), Active(θ))[1][2]
end
forward_dθ(J, θ) = ForwardDiff.derivative(t -> J(plate_sim(t), t), θ)

# the benchmark's `cylinder` in 2D: a circle oscillating across the stream with amplitude A, in a channel
# that is periodic across the stream and has a convective exit, remeasured in every step
# (the closures capture values of the field type only: building a body whose closure holds a Float32 next to
# an Int, i.e. with padding, in the differentiated code fails Enzyme's type analysis in Float32)
function osc(A, L)
    R = oftype(A, L/4); c = SA[oftype(A, L), oftype(A, L)]; ω = oftype(A, 1/L)
    AutoBody((x, _) -> √sum(abs2, x - c) - R, (x, t) -> x - SA[zero(A), A*sin(ω*t)])
end
osc_sim(A; L=16, mem=Array) = Simulation((3L, 2L), (1, 0), L; ν=L/100, body=osc(A, L), T=typeof(A), mem,
    perdir=(2,), exitBC=true)
osc_steps(sim, A) = (sim.body = osc(A, sim.L); measure!(sim); sim_step!(sim, 1e9; max_steps=2, poisson_kwargs=solve_kw(sim));
    energy(sim))
function enzyme_dA(A; kw...)
    sim = osc_sim(A; kw...)
    autodiff(rev, osc_steps, Active, Duplicated(sim, Enzyme.make_zero(sim)), Active(A))[1][2]
end

# 3D: a sphere moving across a channel that is periodic in z, with a convective exit. The body stays in the
# simulation (not set in the cost), so its shadow is the one `Enzyme.make_zero` builds. The gradient with respect
# to the initial velocity is checked along a smooth direction that vanishes near the boundaries, with central
# differences for the convection: the initial flow is uniform, and at the ties of the QUICK limiter (and at u = 0
# on the boundary faces) the derivative depends on how a tie is broken, which differs between Enzyme and
# ForwardDiff, and between Float32 and Float64
function mover(L, T)
    c = SA{T}[L, L/2, L/2]; R = T(L/4)
    AutoBody((x, _) -> √sum(abs2, x - c) - R, (x, t) -> x - SA[zero(t), t/8, zero(t)])
end
mover_sim(T; L=8, mem=Array) = Simulation((2L, L, L), (1, 0, 0), L; ν=L/100, body=mover(L, T), T, mem,
    λ=cds, perdir=(3,), exitBC=true)
function smooth_dir(u; m=3)
    v = zero(u); N = size(u)
    for I ∈ CartesianIndices(v)
        x = Tuple(I)
        all(1 + m .<= x[1:3] .<= N[1:3] .- m) || continue
        v[I] = sin(2π*x[1]/N[1] + 0.3) * cos(2π*x[2]/N[2] + 0.1) * sin(π*x[3]/N[3] + 0.2) * (x[4] == 2 ? 1 : 0.3)
    end
    return v
end
function energy3(sim)
    a = sim.flow
    @loop a.σ[I] = abs2(a.u[I,1]) + abs2(a.u[I,2]) + abs2(a.u[I,3]) over I ∈ inside(a.p)
    sum_view(a.σ, inside(a.p))
end
mover_steps(sim) = (sim_step!(sim, 1e9; max_steps=2, poisson_kwargs=solve_kw(sim)); energy3(sim))
function enzyme_du(T; kw...)
    sim = mover_sim(T; kw...); dsim = Enzyme.make_zero(sim)
    autodiff(rev, mover_steps, Active, Duplicated(sim, dsim))
    return Array(dsim.flow.u)
end
# the derivative along v of the cost after perturbing the initial velocity by ϵv
forward_du(v) = ForwardDiff.derivative(ϵ -> (sim = mover_sim(typeof(ϵ)); sim.flow.u .+= ϵ .* v; mover_steps(sim)), 0.0)

@testset "Enzyme reverse mode" begin
    rng = Xoshiro(1) # own generator: the data must not depend on what else draws random numbers
    # the CPU's Float64 gradients are the reference; every backend is checked against them in its `testfloat`, and the
    # CPU in Float32, the type every backend supports, so that it is checked without a GPU as well
    cases = [(f, f === Array ? Float32 : testfloat(f)) for f ∈ arrays]
    a = rand(rng, 12, 10); s, θ = 1.3, 0.7
    da = zero(a)
    _, _, ds, dθ = autodiff(rev, stencil, Active, Duplicated(a, da), Duplicated(zero(a), zero(a)), Active(s), Active(θ))[1]
    @test da ≈ ForwardDiff.gradient(x -> stencil(x, zero(x), s, θ), a)
    @test ds ≈ ForwardDiff.derivative(x -> stencil(a, zeros(typeof(x), size(a)), x, θ), s)
    @test dθ ≈ ForwardDiff.derivative(x -> stencil(a, zeros(typeof(x), size(a)), s, x), θ)
    @test autodiff(rev, stencil_T, Active, Duplicated(a, zero(a)), Duplicated(zero(a), zero(a)), Active(s), Active(θ))[1][4] ≈ dθ
    if WaterLily.backend == "KernelAbstractions" # (the SIMD backend's loops are differentiated natively)
        @test_throws "not supported in reverse mode" autodiff(rev, mixed, Active, Duplicated(a, zero(a)),
            Duplicated(zero(a), zero(a)), Active(s))
    end
    for (f, T) ∈ cases
        aT = f(T.(a)); daT = zero(aT)
        _, _, dsT, dθT = autodiff(rev, stencil, Active, Duplicated(aT, daT), Duplicated(zero(aT), zero(aT)),
                                  Active(T(s)), Active(T(θ)))[1]
        rtol = tol(T; f64=1e-12, f32=1e-4)
        @test Array(daT) ≈ da rtol=rtol
        @test dsT ≈ ds rtol=rtol
        @test dθT ≈ dθ rtol=rtol
    end

    a = rand(rng, 4, 4, 2); w = [0.3, -1.1]
    da = zero(a); autodiff(rev, reductions, Active, Duplicated(a, da), Const(w))
    da_ref = zero(a); da_ref[argmax(a)] += 2; da_ref .+= reshape(w, 1, 1, 2)
    @test da ≈ da_ref
    for (f, T) ∈ cases
        aT = f(T.(a)); daT = zero(aT)
        autodiff(rev, reductions, Active, Duplicated(aT, daT), Const(T.(w)))
        @test Array(daT) ≈ da_ref rtol=tol(T; f64=1e-14, f32=1e-6)
    end

    for (perdir, masked) ∈ (((), false), ((1,), false), ((), true)) # Neumann, periodic, masked cells
        p, w, act = poisson_case(rng, perdir, masked)
        z = copy(p.z); dp = Enzyme.make_zero(p)
        autodiff(rev, poisson_J, Active, Duplicated(p, dp), Const(w))
        A, idx = poisson_dense(p, act)
        @test dp.z[idx] ≈ A \ w[idx] rtol=1e-8 # z̄ = A⁺w
        @test all(iszero, dp.z[.!act])
        for (I, i) ∈ ((CartesianIndex(5, 6), 1), (CartesianIndex(12, 4), 2), (CartesianIndex(9, 13), 1))
            p.L[I, i] == 0 && continue
            @test dp.L[I, i] ≈ poisson_dJdL(p, w, act, z, I, i) rtol=1e-8
        end
    end

    θ₀ = π/36
    g1 = enzyme_dθ(one_step, θ₀)
    @test g1 ≈ forward_dθ(one_step, θ₀) rtol=1e-6
    g3 = enzyme_dθ(three_steps, θ₀)
    @test g3 ≈ forward_dθ(three_steps, θ₀) rtol=1e-6
    if v"1.11" <= VERSION < v"1.12" # a cost accumulated by evolve! hits EnzymeAD/Enzyme.jl#3322 on Julia 1.11
        @test_broken enzyme_dθ(running_cost, θ₀) ≈ forward_dθ(running_cost, θ₀) rtol=1e-6
        @test_broken enzyme_dθ(running_force, θ₀) ≈ forward_dθ(running_force, θ₀) rtol=1e-6
    else
        @test enzyme_dθ(running_cost, θ₀) ≈ forward_dθ(running_cost, θ₀) rtol=1e-6
        @test enzyme_dθ(running_force, θ₀) ≈ forward_dθ(running_force, θ₀) rtol=1e-6
    end
    @test_throws "make the step function type-stable" enzyme_dθ(unstable_cost, θ₀)
    let r₀ = 0.5, sim = circle_sim(r₀) # a parameter inside the sdf; a cost not invariant to the pressure's constant
        gr = autodiff(rev, drag_step, Active, Duplicated(sim, Enzyme.make_zero(sim)), Active(r₀))[1][2]
        @test gr ≈ ForwardDiff.derivative(r -> drag_step(circle_sim(r), r), r₀) rtol=1e-6
    end
    for (f, T) ∈ cases # one step, and three through evolve!
        rtol = tol(T; f64=1e-8, f32=1e-3)
        @test enzyme_dθ(one_step, T(θ₀); mem=f) ≈ g1 rtol=rtol
        @test enzyme_dθ(three_steps, T(θ₀); mem=f) ≈ g3 rtol=rtol
    end

    # a moving body remeasured in every step, periodic and exit boundaries: 2D, with respect to the motion
    A₀ = 2.0
    gA = enzyme_dA(A₀)
    @test gA ≈ ForwardDiff.derivative(a -> osc_steps(osc_sim(a), a), A₀) rtol=1e-6
    for (f, T) ∈ cases
        @test enzyme_dA(T(A₀); mem=f) ≈ gA rtol=tol(T; f64=1e-8, f32=1e-3)
    end
    # and 3D, with respect to the initial velocity (along a direction v)
    du = enzyme_du(Float64)
    v = smooth_dir(du)
    @test sum(du .* v) ≈ forward_du(v) rtol=1e-6
    for (f, T) ∈ cases # (in Float32, the solves to 1e-5 leave 1.5e-3 here)
        @test sum(enzyme_du(T; mem=f) .* v) ≈ sum(du .* v) rtol=tol(T; f64=1e-8, f32=5e-3)
    end
end
