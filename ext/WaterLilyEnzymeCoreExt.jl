module WaterLilyEnzymeCoreExt

using WaterLily
using WaterLily: launch!, itemarg, sum_view, max_all, sum_dims
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

end # module
