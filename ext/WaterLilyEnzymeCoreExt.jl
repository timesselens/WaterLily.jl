module WaterLilyEnzymeCoreExt

using WaterLily
using WaterLily: launch!, itemarg, sum_view, max_all, sum_dims, solver!, update!, @loop, inside, δ, CIj, slice,
    evolve!, sim_time, sim_info, accumulate_cost, AbstractSimulation
using EnzymeCore
using EnzymeCore.EnzymeRules
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

# Reverse-mode rules for the device work of a WaterLily time step. Enzyme differentiates the
# host code; the device work goes through `@loop` (→ `launch!`), three reductions and the
# pressure solve, each with a rule below, so the same rules serve the CPU and every GPU backend.

## `@loop` kernels
#
# `launch!(f, R, args...)` runs the item function `f(I, args...)` over `R`. `f` is handed to Enzyme as an
# argument of `_call` (Enzyme has no `Active` function annotation), so a closure that captures active values,
# e.g. `measure!`'s `fill!` capturing the body, gets a derivative.
# - CPU: the launch is differentiated as one serial loop (`_serial!`), natively, as the SIMD backend's loops.
# - GPU: Enzyme runs on `f` item by item: an augmented kernel stores one tape per item, a reverse kernel runs
#   the item adjoints. `Active` arguments (scalars such as `dt`, and such closures) return one cotangent per
#   item; these are collected in a buffer and summed.

# One method per number of arguments, like `launch!`'s kernels: a Vararg would gather the arguments in
# one tuple on the stack, which Enzyme cannot type and must treat as active as a whole (it holds the active
# scalars too), so the integers loaded back from it (sizes, indices) would count as active.
for n in 0:16
    a = [Symbol(:a, k) for k in 1:n]; A = [Symbol(:A, k) for k in 1:n]
    @eval @inline _call(f::F, I, $((:($x::$X) for (x, X) in zip(a, A))...)) where {F,$(A...)} =
        (f(I, $((:(itemarg($x)) for x in a)...)); nothing)
end

# item thunk arguments are (f, I, args...); ModifiedBetween also has a leading entry for `_call`
_modified(config) = (false, overwritten(config)[2], false, overwritten(config)[4:end]...)
_mode(config) = EnzymeCore.set_runtime_activity(ReverseSplitModified(ReverseSplitWithPrimal, Val(_modified(config))), config)

# GPU: the thunks are generated in the kernel (deferred)
@kernel function _aug_gpu!(tape, mode, ::Val{TT}, f, @Const(I0), args...) where {TT}
    I = @index(Global, Cartesian); i = @index(Global, Linear)
    fwd, _ = EnzymeCore.autodiff_deferred_thunk(mode, TT, Const{typeof(_call)}, Const{Nothing},
        Core.Typeof(f), Const{typeof(I)}, map(Core.Typeof, args)...)
    @inbounds tape[i] = fwd(Const(_call), f, Const(I + I0), args...)[1]
end
@kernel function _rev_gpu!(tape, dbufs, mode, ::Val{TT}, f, @Const(I0), args...) where {TT}
    I = @index(Global, Cartesian); i = @index(Global, Linear)
    _, rev = EnzymeCore.autodiff_deferred_thunk(mode, TT, Const{typeof(_call)}, Const{Nothing},
        Core.Typeof(f), Const{typeof(I)}, map(Core.Typeof, args)...)
    d = rev(Const(_call), f, Const(I + I0), args..., @inbounds(tape[i]))[1]
    _store!(dbufs, d, i)
end
# CPU: the launch as one serial loop, which Enzyme differentiates natively like the loops of the SIMD backend, with
# one tape per launch and the cotangents of `Active` arguments from Enzyme. Per-item thunks run from KernelAbstractions'
# CPU kernels cost 100x more: dynamic calls (a Vararg kernel is not specialized) and a tape per item.
# One method per number of arguments, as `_call`. `f` is inlined into the loop: a call makes Enzyme keep a tape per
# item (9x slower, 70x the memory). `@inbounds` covers the loop: with bounds checks in `@simd`'s index arithmetic,
# Enzyme keeps each array's shadow `Memory` on the tape once per outer iteration instead of once.
for n in 0:16
    a = [Symbol(:a, k) for k in 1:n]; A = [Symbol(:A, k) for k in 1:n]
    @eval function _serial!(f::F, R, $((:($x::$X) for (x, X) in zip(a, A))...)) where {F,$(A...)}
        @inbounds @simd for I in R
            @inline f(I, $((:(itemarg($x)) for x in a)...))
        end
        return nothing
    end
end
# the derivative of an argument as Enzyme requires it (`T` for `Active{T}`, else `nothing`): the tape is untyped
_dret(d, ::Active{T}) where {T} = d::T
_dret(d, a) = nothing
# thunk arguments are (f, R, args...), as for launch!. The rule makes them through `invokelatest`: made while Enzyme
# compiles the rule into a caller, Enzyme failed on `sincos(::Float64)` in the item function (a moving body in Float64).
_serial_mode(config) = EnzymeCore.set_runtime_activity(
    ReverseSplitModified(ReverseSplitWithPrimal, Val((false, Base.tail(overwritten(config))...))), config)
_serial_thunks(config, f, R, args) = EnzymeCore.autodiff_thunk(_serial_mode(config), Const{typeof(_serial!)},
    Const{Nothing}, Core.Typeof(f), Core.Typeof(R), map(Core.Typeof, args)...)

# The cotangent of an Active argument (a number, or a struct such as a closure) is stored per item
# as its floating-point fields, one column of a (k × items) buffer, and summed over the items.
@generated function _leaves(x::T) where {T}
    exprs = Any[]
    function walk(ex, S)
        if S <: AbstractFloat
            push!(exprs, ex)
        elseif isstructtype(S)
            for i in 1:fieldcount(S)
                walk(:(getfield($ex, $i)), fieldtype(S, i))
            end
        end
    end
    walk(:x, T)
    return :(Base.@_inline_meta; ($(exprs...),))
end
# the derivative of `x`: its floating-point fields taken, in order, from `v`, other numbers zero
# (as in the derivatives Enzyme builds itself)
@generated function _rebuild(x::T, v) where {T}
    k = Ref(0)
    function build(ex, S)
        S <: AbstractFloat && return :(convert($S, v[$(k[] += 1)]))
        S <: Number && return :(zero($S))
        (isstructtype(S) && fieldcount(S) > 0) || return ex
        fields = [build(:(getfield($ex, $i)), fieldtype(S, i)) for i in 1:fieldcount(S)]
        return S <: Tuple ? Expr(:tuple, fields...) : Expr(:new, S, fields...)
    end
    return build(:x, T)
end
function _leaftypes(S)
    S <: AbstractFloat && return Any[S]
    isstructtype(S) || return Any[]
    return reduce(vcat, (_leaftypes(fieldtype(S, i)) for i in 1:fieldcount(S)); init = Any[])
end

# the buffer of an Active argument, in the promoted type of its fields; none if it has no floating-point fields
_buffer(backend, a::Active, n) =
    (L = _leaftypes(typeof(a.val)); isempty(L) ? nothing : KA.allocate(backend, promote_type(L...), (length(L), n)))
_buffer(backend, a, n) = nothing
# the derivative of each argument, typed as Enzyme requires (`T` for `Active{T}`, else `nothing`)
_reduce(b, a::Active{T}) where {T} = _rebuild(a.val, Array(vec(sum(b; dims = 2))))::T
_reduce(::Nothing, a::Active{T}) where {T} = _rebuild(a.val, ())::T
_reduce(b, a) = nothing

@inline _store!(::Tuple{}, ::Tuple{}, i) = nothing
# Enzyme returns the derivatives as a NamedTuple when one of them holds a Julia object, e.g. a closure
# that captures a type
@inline _store!(dbufs::Tuple, d::NamedTuple, i) = _store!(dbufs, values(d), i)
@inline function _store!(dbufs::Tuple, d::Tuple, i)
    _store1!(first(dbufs), first(d), i)
    _store!(Base.tail(dbufs), Base.tail(d), i)
end
@inline _store1!(::Nothing, x, i) = nothing
@inline _store1!(b, x, i) = _storeleaves!(b, i, 1, _leaves(x))
@inline _storeleaves!(b, i, j, ::Tuple{}) = nothing
@inline function _storeleaves!(b, i, j, t::Tuple)
    @inbounds b[j, i] = first(t)
    _storeleaves!(b, i, j + 1, Base.tail(t))
end

_check(a) = a isa MixedDuplicated &&
    error("WaterLily @loop: an argument holding both arrays and active values ($(typeof(a.val))) is not supported in reverse mode")

function _tape_type(backend::KA.GPU, mode, targs)
    job = EnzymeCore.compiler_job_from_backend(backend, typeof(() -> return), Tuple{})
    dev = map(a -> KA.argconvert(_aug_gpu!(backend, 64), a), targs) # device-side argument types
    EnzymeCore.tape_type(job, mode, Const{typeof(_call)}, Const{Nothing}, map(Core.Typeof, dev)...)
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(launch!)}, ::Type{RT},
        f::Annotation, R::Const, args::Vararg{Annotation,N}) where {RT,N}
    foreach(_check, (f, args...))
    backend = KA.get_backend(first(args).val)
    if !(backend isa KA.GPU)
        fwd, _ = Base.invokelatest(_serial_thunks, config, f, R, args)
        return AugmentedReturn{Nothing,Nothing,Any}(nothing, nothing, fwd(Const(_serial!), f, R, args...)[1])
    end
    I0 = R.val[1] - oneunit(R.val[1])
    mode = _mode(config)
    TT = _tape_type(backend, mode, (f, Const(I0), args...))
    tape = KA.allocate(backend, TT, length(R.val))
    _aug_gpu!(backend, 64)(tape, mode, Val(TT), f, I0, args...; ndrange = size(R.val))
    return AugmentedReturn{Nothing,Nothing,Any}(nothing, nothing, (tape, mode, TT))
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(launch!)}, ::Type{RT}, tape,
        f::Annotation, R::Const, args::Vararg{Annotation,N}) where {RT,N}
    backend = KA.get_backend(first(args).val)
    if !(backend isa KA.GPU)
        _, rev = Base.invokelatest(_serial_thunks, config, f, R, args)
        d = rev(Const(_serial!), f, R, args..., tape)[1]
        return (_dret(d[1], f), nothing, map((a, k) -> _dret(d[k], a), args, ntuple(i -> i + 2, Val(N)))...)
    end
    (tape, mode, TT) = tape
    I0 = R.val[1] - oneunit(R.val[1])
    dbufs = map(a -> _buffer(backend, a, length(R.val)), (f, Const(I0), args...))
    _rev_gpu!(backend, 64)(tape, dbufs, mode, Val(TT), f, I0, args...; ndrange = size(R.val))
    return (_reduce(dbufs[1], f), nothing, map(_reduce, Base.tail(Base.tail(dbufs)), args)...)
end

## reductions
#
# Enzyme checks the *inferred* return type of an augmented rule, and GPU reductions do not always
# infer (e.g. `findmax` on an MtlArray), so each primal value is asserted to `eltype(RT)`.

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(sum_view)}, ::Type{RT},
        a::Annotation, I::Vararg{Const,N}) where {RT,N}
    s = sum_view(a.val, map(i -> i.val, I)...)::eltype(RT)
    return AugmentedReturn(needs_primal(config) ? s : nothing, nothing, nothing)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(sum_view)}, dret::Active, tape,
        a::Annotation, I::Vararg{Const,N}) where {N}
    a isa Const || (@view(a.dval[map(i -> i.val, I)...]) .+= dret.val)
    return (nothing, ntuple(_ -> nothing, Val(N))...)
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(max_all)}, ::Type{RT},
        a::Annotation) where {RT}
    m, i = findmax(a.val)
    return AugmentedReturn(needs_primal(config) ? m::eltype(RT) : nothing, nothing, LinearIndices(a.val)[i]::Int)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(max_all)}, dret::Active, i,
        a::Annotation)
    a isa Const || (@view(vec(a.dval)[i:i]) .+= dret.val) # one-hot at the argmax, without scalar indexing
    return (nothing,)
end

# `sum_dims` returns a host `Vector`, so its cotangent arrives in the shadow vector made here
function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(sum_dims)}, ::Type{RT},
        T::Const, a::Annotation, n::Const) where {RT}
    v = sum_dims(T.val, a.val, n.val)::eltype(RT)
    dv = zero(v)
    return AugmentedReturn(needs_primal(config) ? v : nothing, needs_shadow(config) ? dv : nothing, dv)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(sum_dims)}, ::Type{RT}, dv,
        T::Const, a::Annotation, ::Const{Val{n}}) where {RT,n}
    if !(a isa Const)
        ā = a.dval
        v̄ = copyto!(similar(ā, eltype(ā), length(dv)), dv) # to the device of `a`
        ā .+= reshape(v̄, ntuple(i -> i <= n ? 1 : size(ā, i), Val(ndims(ā))))
    end
    return (nothing, nothing, nothing)
end

## multigrid coefficients
#
# The coarse levels of a `MultiLevelPoisson` only precondition the solve: the solution depends on the fine
# coefficients alone, and the implicit-function rule on `solver!` (below) gives them no cotangent. So the
# restriction of the coefficients to the coarse levels (`update!` after `measure!` for moving bodies) has
# a zero derivative, and Enzyme does not need to differentiate its loops.
EnzymeRules.inactive(::typeof(WaterLily.restrictL!), args...) = nothing

## pressure Poisson solve
#
# `solver!` iterates to a tolerance, so rather than differentiating the iterations the rule uses the
# implicit function theorem on the solution of A(L,D) x = z, which `pin_pressure!` makes unique (zero mean
# over the active cells, zero on the inactive ones). A is symmetric (the face coefficient L[I,i] couples I and
# I-δᵢ in both rows), so λ = A⁺x̄ comes from the same solver, pinned the same way, run in a separate workspace
# so the primal arrays keep their values for the rest of the reverse pass. x̄ is projected like x first, so the
# adjoint system has a solution. Then z̄ += λ and L̄, D̄ -= λxᵀ on A's sparsity pattern; the initial guess,
# residual and multigrid scratch arrays get no cotangent. The adjoint solve stops at the relative residual the
# primal solve reached (so a tighter `tol` in `poisson_kwargs` gives both).

const _Poissons = Union{WaterLily.Poisson,WaterLily.MultiLevelPoisson}
# Enzyme may pass a solver as MixedDuplicated (shadow in a Ref)
const _PoissonArg = Union{Duplicated{<:_Poissons},MixedDuplicated{<:_Poissons}}
_shadow(p::Duplicated) = p.dval
_shadow(p::MixedDuplicated) = p.dval[]
_fine(p::WaterLily.Poisson) = p
_fine(p::WaterLily.MultiLevelPoisson) = first(p.levels)

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(solver!)}, ::Type{RT},
        p::_PoissonArg; kwargs...) where {RT}
    zmax = maximum(abs, p.val.z)
    ret = func.val(p.val; kwargs...)
    rtol = WaterLily.L∞(_fine(p.val)) / max(zmax, floatmin(zmax)) # the relative residual the solve reached
    tape = (copy(p.val.x), copy(p.val.L), rtol) # L (= μ₀) is overwritten by the next measure!
    return AugmentedReturn(needs_primal(config) ? ret : nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(solver!)}, ::Type{RT}, (x, L, rtol),
        p::_PoissonArg; kwargs...) where {RT}
    P, dP = p.val, _shadow(p)
    x̄ = _fine(dP).x
    _perBC_adjoint!(x̄, P.perdir) # the solve ends by copying periodic ghosts from the interior
    ws = _workspace(P)
    copyto!(ws.L, L); update!(ws)
    # the part of x̄ in A's range, projected as the solve pins x. Without it a constant can remain on the active
    # cells when some are inactive (`residual!` removes the mean over all interior cells), and the solve turns it
    # into errors in λ
    copyto!(ws.x, x̄); WaterLily.pin_pressure!(_fine(ws)); copyto!(ws.z, ws.x)
    x̄max = maximum(abs, @view(ws.z[inside(ws.z)]))
    if x̄max > 0
        fill!(ws.x, 0)
        # adjoint solved to the relative accuracy the primal solve reached; other keywords (e.g. `itmx`) as given
        solver!(ws; kwargs..., tol = clamp(rtol, 10eps(eltype(x)), 1e-2) * x̄max) # λ = ws.x, pinned
        fdP = _fine(dP)
        _accumulate!(fdP.z, fdP.L, fdP.D, ws.x, x)
    end
    _zero_scratch!(dP)
    return (nothing,)
end

# adjoint solver with its own arrays, cached per primal solver. The cache is keyed by the identity of the
# solution array (a WeakKeyDict would hash GPU arrays by value, element by element) and holds it weakly:
# workspaces of collected solvers are dropped when the next one is made.
const _workspaces = Dict{UInt,Tuple{WeakRef,Any}}()
const _workspaces_lock = ReentrantLock()
function _workspace(P)
    lock(_workspaces_lock) do
        entry = get(_workspaces, objectid(P.x), nothing)
        entry !== nothing && entry[1].value === P.x && return entry[2]
        filter!(kv -> last(kv)[1].value !== nothing, _workspaces)
        ws = _new_workspace(P)
        _workspaces[objectid(P.x)] = (WeakRef(P.x), ws)
        return ws
    end
end
_new_workspace(P::WaterLily.MultiLevelPoisson) = WaterLily.MultiLevelPoisson(zero(P.x), copy(P.L), zero(P.z);
    perdir = P.perdir, maxlevels = length(P.levels) - 1)
_new_workspace(P::WaterLily.Poisson) = WaterLily.Poisson(zero(P.x), copy(P.L), zero(P.z); perdir = P.perdir)

# transpose of `perBC!` (ghost ← interior copies): ghost cotangents go back to their source cells
_perBC_adjoint!(a, ::Tuple{}) = nothing
function _perBC_adjoint!(a, perdir, N = size(a))
    for j in reverse(perdir)
        @loop (a[CIj(j, I, 2)] += a[I]; a[I] = zero(eltype(a))) over I ∈ slice(N, N[j], j)
        @loop (a[CIj(j, I, N[j] - 1)] += a[I]; a[I] = zero(eltype(a))) over I ∈ slice(N, 1, j)
    end
end

# z̄ += λ; L̄, D̄ -= λxᵀ where (Ax)[I] = D[I]x[I] + Σᵢ L[I,i]x[I-δᵢ] + L[I+δᵢ,i]x[I+δᵢ]
function _accumulate!(z̄, L̄, D̄, λ, x)
    @loop z̄[I] += λ[I] over I ∈ inside(λ)
    for i in 1:ndims(λ)
        @loop L̄[I, i] -= λ[I] * x[I - δ(i, I)] over I ∈ inside(λ)
        @loop L̄[I + δ(i, I), i] -= λ[I] * x[I + δ(i, I)] over I ∈ inside(λ)
    end
    @loop D̄[I] -= λ[I] * x[I] over I ∈ inside(λ)
end

# the solution does not depend on the interior initial guess, nor on the scratch arrays
_zero_inside!(a) = @loop a[I] = zero(eltype(a)) over I ∈ inside(a)
function _zero_scratch!(dp::WaterLily.Poisson)
    _zero_inside!(dp.x); fill!(dp.r, 0); fill!(dp.ϵ, 0)
    return nothing
end
function _zero_scratch!(dp::WaterLily.MultiLevelPoisson)
    _zero_scratch!(first(dp.levels))
    for l in dp.levels[2:end]
        fill!(l.x, 0); fill!(l.r, 0); fill!(l.ϵ, 0); fill!(l.z, 0)
    end
    return nothing
end

## time loop: `evolve!` with one checkpoint per step
#
# Differentiating `evolve!`'s loop directly would keep every value Enzyme needs from every step. Instead
# the forward pass stores what a step reads before writing it (u, p, σ and the number of time steps) and
# the reverse pass replays the steps backwards: restore the checkpoint, run the step as a split-mode thunk,
# seed the time step Δt it appended with its cotangent (accumulated by the later steps), run its reverse.
# The flow is left as it was before `evolve!`.

_apply(f::F, s) where {F} = f(s)

_checkpoint(s) = (copy(s.flow.u), copy(s.flow.p), copy(s.flow.σ), length(s.flow.Δt))
function _restore!(s, (u, p, σ, n))
    copyto!(s.flow.u, u); copyto!(s.flow.p, p); copyto!(s.flow.σ, σ)
    resize!(s.flow.Δt, n)
    return s
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(evolve!)}, ::Type{RT},
        f::Annotation, sim::Duplicated{<:AbstractSimulation}, t_end::Annotation;
        max_steps = typemax(Int), verbose = false, init = nothing) where {RT}
    # a cost that doesn't infer reaches the rule as `Duplicated{Any}`, which the checkpointed replay can't seed
    RT <: Const || isconcretetype(eltype(RT)) || error("WaterLily evolve!: the cost of the step function is inferred ",
        "as `$(eltype(RT))`, so Enzyme can't tell whether it is active; make the step function type-stable. For ",
        "example, `sim_time(sim)` is not inferred (`Simulation`'s `U` and `L` are abstract): use ",
        "`WaterLily.time(sim.flow)*U/L` with `U` and `L` of a concrete type")
    s, ds = sim.val, sim.dval
    steps₀ = length(s.flow.Δt); snaps = Any[]; J = init
    while sim_time(s) < t_end.val && length(s.flow.Δt) - steps₀ < max_steps
        push!(snaps, _checkpoint(s))
        J = accumulate_cost(J, f.val(s))
        verbose && sim_info(s)
    end
    resize!(ds.flow.Δt, length(s.flow.Δt)) # the shadow follows the appended time steps (as zeros)
    ds.flow.Δt[steps₀+1:end] .= 0
    return AugmentedReturn(needs_primal(config) ? J : nothing, nothing, snaps)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(evolve!)}, dret, snaps,
        f::Annotation, sim::Duplicated{<:AbstractSimulation}, t_end::Annotation; kwargs...)
    dt_end = t_end isa Active ? zero(t_end.val) : nothing
    isempty(snaps) && return (nothing, nothing, dt_end)
    s, ds = sim.val, sim.dval
    f isa Const || error("WaterLily evolve!: a step function with active captured values is not supported in reverse mode")
    J̄ = dret isa Active ? dret.val : nothing
    mode = EnzymeCore.set_runtime_activity(ReverseSplitWithPrimal, config)
    fwd, rev = EnzymeCore.autodiff_thunk(mode, Const{typeof(_apply)}, J̄ === nothing ? Const : Active,
        Core.Typeof(f), Duplicated{typeof(s)})
    for k in length(snaps):-1:1
        n = snaps[k][4]
        c = ds.flow.Δt[n+1] # cotangent of the Δt appended by step k
        _restore!(s, snaps[k]); resize!(ds.flow.Δt, n)
        tape, _, _ = fwd(Const(_apply), f, sim)
        ds.flow.Δt[end] = c # set, not add: the slot the forward pass appended to the shadow is not zeroed
        J̄ === nothing ? rev(Const(_apply), f, sim, tape) : rev(Const(_apply), f, sim, J̄, tape)
    end
    _restore!(s, snaps[1]); resize!(ds.flow.Δt, snaps[1][4]) # the state before evolve!
    return (nothing, nothing, dt_end)
end

end # module
