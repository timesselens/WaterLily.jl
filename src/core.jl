using KernelAbstractions: get_backend, @index, @kernel, @Const, supports_float64
using LoggingExtras

# custom log macro
const _psolver = Logging.LogLevel(-123) # custom log level for pressure solver, needs the negative sign
macro log(exs...)
    quote
        @logmsg _psolver $(map(x -> esc(x), exs)...)
    end
end
"""
    logger(fname="WaterLily")

Set up a logger to write the pressure solver data to a logging file named `WaterLily.log`.
"""
function logger(fname::String="WaterLily")
    ENV["JULIA_DEBUG"] = all
    logger = FormatLogger(ifelse(fname[end-3:end]==".log",fname[1:end-4],fname)*".log"; append=false) do io, args
        args.level == _psolver && print(io, args.message)
    end;
    global_logger(logger);
    # put header in file
    @log "p/c, iter, r∞, r₁, ω\n"
end

@inline CI(a...) = CartesianIndex(a...)
"""
    CIj(j,I,k)
Replace jᵗʰ component of CartesianIndex with k
"""
CIj(j,I::CartesianIndex{d},k) where d = CI(ntuple(i -> i==j ? k : I[i], d))

"""
    δ(i,::Val{N})
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a;buff=1)

Return CartesianIndices range excluding a single layer of cells on all boundaries.
"""
@inline inside(a::AbstractArray;buff=1) = CartesianIndices(map(ax->first(ax)+buff:last(ax)-buff,axes(a)))

"""
    inside_u(dims,j)

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _vector_ array on face `j` with size `dims`.
"""
function inside_u(dims::NTuple{N},j) where {N}
    CartesianIndices(ntuple( i-> i==j ? (3:dims[i]-1) : (2:dims[i]), N))
end
@inline inside_u(dims::NTuple{N}) where N = CartesianIndices((map(i->(2:i-1),dims)...,1:N))
@inline inside_u(u::AbstractArray) = CartesianIndices(map(i->(2:i-1),size(u)[1:end-1]))
splitn(n) = Base.front(n),last(n)
size_u(u) = splitn(size(u))

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example,

    @inside p[I] = sum(loc(0,I))

becomes

    @loop p[I] = sum(loc(0,I)) over I ∈ inside(p)

See [`@loop`](@ref).
"""
macro inside(ex)
    # Make sure it's a single assignment
    @assert ex.head == :(=) && ex.args[1].head == :(ref)
    a,I = ex.args[1].args[1:2]
    return quote # loop over the size of the reference
        WaterLily.@loop $ex over $I ∈ inside($a)
    end |> esc
end

# Could also use ScopedValues in Julia 1.11+
using Preferences
const backend = @load_preference("backend", "KernelAbstractions")
function set_backend(new_backend::String)
    if !(new_backend in ("SIMD", "KernelAbstractions"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

"""
    launch!(f, R, args...)

Run `f(I, args...)` for every `I ∈ R` in a KernelAbstractions kernel on the backend of `first(args)`.
This is what [`@loop`](@ref) expands to on the KernelAbstractions backend: `f` is the loop body and
`args` are the arrays and values it uses. The loops share a few module-level kernels (one per number
of arguments, up to 16), so each loop is a plain function of `I` that AD tools can differentiate item
by item. An argument `TypeArg{T}()` reaches `f` as the type `T`.
"""
function launch!(f::F, R, args::Vararg{Any,N}) where {F,N}
    loop_kernel(Val(N))(get_backend(first(args)),64)(f,R[1]-oneunit(R[1]),args...,ndrange=size(R))
    return nothing
end
# `@loop` wraps type arguments (e.g. `T` in `zero(T)`) at the call site, where `T` is known: in a kernel's
# argument tuple a type is a `DataType`, and the CPU backend then calls the kernel dynamically per workgroup.
struct TypeArg{T} end
kernarg(::Type{T}) where T = TypeArg{T}()
kernarg(a) = a
@inline itemarg(::TypeArg{T}) where T = T
@inline itemarg(a) = a
# One kernel per number of arguments, each argument separate and typed as in a hand-written kernel:
# a Vararg or Tuple of arguments makes the CPU kernels slower, and untyped arguments that are only
# passed on (e.g. the convection scheme `λ`) are not specialized. The loop body is inlined.
for n in 1:16
    a = [Symbol(:a,k) for k in 1:n]; A = [Symbol(:A,k) for k in 1:n]; kern = Symbol(:_loop,n,:!)
    @eval begin
        @kernel function $kern(f::F,@Const(I0),$((:($x::$X) for (x,X) in zip(a,A))...)) where {F,$(A...)}
            I = @index(Global,Cartesian)
            @inline f(I+I0,$((:(itemarg($x)) for x in a)...))
        end
        loop_kernel(::Val{$n}) = $kern
    end
end
loop_kernel(::Val{N}) where N = throw(ArgumentError("@loop: the loop body uses $N variables, at most 16 are supported"))

"""
    @loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running in serial,
or KernelAbstractions when running multi-threaded CPU or GPU.

For example

    @loop a[I,i] += sum(loc(i,I)) over I ∈ R

becomes

    @inbounds @simd for I ∈ R
        @fastmath @inbounds a[I,i] += sum(loc(i,I))
    end

on serial execution, or

    launch!((I,a,i) -> (@fastmath @inbounds a[I,i] += sum(loc(i,I)); nothing), R, a, i)

when multi-threading on CPU or using CuArrays. See [`launch!`](@ref).
Note that `get_backend` is used on the _first_ variable in `expr` (`a` in this example).
"""
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args
    sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    symT = [gensym() for _ in 1:length(sym)] # generate a list of types for each symbol
    symWtypes = joinsymtype(rep.(sym),symT) # symbols with types: [a::A, b::B, ...], also forces specialization on functions
    @static if backend == "KernelAbstractions"
        return quote # the loop body becomes an item function of I and the grabbed arguments
            $launch!(function ($I,$(symWtypes...)) where {$(symT...)}
                @fastmath @inbounds $ex
                return nothing
            end, $R, $((:($kernarg($s)) for s in sym)...))
        end |> esc
    else # backend == "SIMD"
        @gensym(kern, r) # generate unique names for the kernel function and the range
        return quote
            function $kern($(symWtypes...)) where {$(symT...)}
                $r = $R # `@inbounds` covers the loop, including `@simd`'s index arithmetic, but not the range
                @inbounds @simd for $I ∈ $r
                    @fastmath @inbounds $ex
                end
            end
            $kern($(sym...))
        end |> esc
    end
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])          # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex
joinsymtype(sym::Symbol,symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym,symT) = zip(sym,symT) .|> x->joinsymtype(x...)

using StaticArrays
"""
    loc(i,I,T) = loc(Ii,T)

Location in space of the cell at CartesianIndex `I` at face `i`.
Using `i=0` returns the cell center s.t. `loc = I`. `T` sets the return number type.
"""
@inline loc(i,I::CartesianIndex{N},T=Float32) where N = SVector{N,T}(I.I .- T(1.5) .- δ(i,I).I ./T(2))
@inline loc(Ii::CartesianIndex,T=Float32) = loc(last(Ii),Base.front(Ii),T)
Base.last(I::CartesianIndex) = last(I.I)
Base.front(I::CartesianIndex) = CI(Base.front(I.I))
"""
    slice(dims,i,j,low=1)

Return `CartesianIndices` range slicing through an array of size `dims` in
dimension `j` at index `i`. `low` optionally sets the lower extent of the range
in the other dimensions.
"""
function slice(dims::NTuple{N},i,j,low=1) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:dims[k]), N))
end

"""
    BC!(a,U,saveexit=false,perdir=(),t=0)

Apply boundary conditions to the ghost cells of a _vector_ field. A Dirichlet
condition `a[I,i]=U[i]` is applied to the vector component _normal_ to the domain
boundary. For example `aₓ(x)=Aₓ ∀ x ∈ minmax(X)`.
A zero Neumann condition is applied to the tangential components.
`saveexit=true` leaves the normal component on the `x` exit face (upper `x` boundary) untouched,
so a convective outflow set by `exitBC!` is kept and every other face is always overwritten.
`perdir` lists the dimensions using periodic BCs instead.
`t` is passed to `U` when it is a function `(i,x,t)->...`.
"""
BC!(a,U,saveexit=false,perdir=(),t=0) = BC!(a,(i,x,t)->U[i],saveexit,perdir,t)
function BC!(a,uBC::Function,saveexit=false,perdir=(),t=0)
    N,n = size_u(a)
    for i ∈ 1:n, j ∈ 1:n
        if j in perdir
            @loop a[I,i] = a[CIj(j,I,N[j]-1),i] over I ∈ slice(N,1,j)
            @loop a[I,i] = a[CIj(j,I,2),i] over I ∈ slice(N,N[j],j)
        else
            if i==j # Normal direction, Dirichlet
                for s ∈ (1,2)
                    @loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,s,j)
                end
                (!saveexit || i>1) && (@loop a[I,i] = uBC(i,loc(i,I),t) over I ∈ slice(N,N[j],j)) # overwrite exit
            else    # Tangential directions, Neumann
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I+δ(j,I),i]-uBC(i,loc(i,I+δ(j,I)),t) over I ∈ slice(N,1,j)
                @loop a[I,i] = uBC(i,loc(i,I),t)+a[I-δ(j,I),i]-uBC(i,loc(i,I-δ(j,I)),t) over I ∈ slice(N,N[j],j)
            end
        end
    end
end

"""
    exitBC!(u,u⁰,Δt)

Apply a 1D convection scheme to fill the ghost cell on the exit of the domain.
"""
function exitBC!(u,u⁰,Δt)
    N,_ = size_u(u)
    exitR = slice(N.-1,N[1],1,2)              # exit slice excluding ghosts
    U = sum(@view(u[slice(N.-1,2,1,2),1]))/length(exitR) # inflow mass flux
    @loop u[I,1] = u⁰[I,1]-U*Δt*(u⁰[I,1]-u⁰[I-δ(1,I),1]) over I ∈ exitR
    ∮u = sum(@view(u[exitR,1]))/length(exitR)-U   # mass flux imbalance
    @loop u[I,1] -= ∮u over I ∈ exitR         # correct flux
end
"""
    perBC!(a,perdir)

Apply periodic conditions to the ghost cells of a _scalar_ field.
"""
perBC!(a,::Tuple{}) = nothing
perBC!(a, perdir, N = size(a)) = for j ∈ perdir
    @loop a[I] = a[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
    @loop a[I] = a[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
end

using ForwardDiff
using ForwardDiff: Dual, partials, Tag

# Inner-derivative tag for measure's gradient/jacobian/derivative. `≺` is
# overloaded so it always ranks newer than any `ForwardDiff.Tag`, folding the
# precedence comparison at compile time and sidestepping `tagcount` (order-
# sensitive on GPU codegen, the original cause of nested-FD crashes in kernels).
struct _InnerTag end
@inline ForwardDiff.:≺(::Type{<:Tag}, ::Type{_InnerTag}) = true
@inline ForwardDiff.:≺(::Type{_InnerTag}, ::Type{<:Tag}) = false
@inline ForwardDiff.:≺(::Type{_InnerTag}, ::Type{_InnerTag}) = false

# Tag-aware partial extractor. The fallback returns zero when `y` is not an
# `_InnerTag` dual — `f` did not depend on the seeded input so the inner
# derivative is exactly zero. Without it, an outer-tag `Dual` (from closure
# capture) would silently leak its outer partial.
@inline _ip(y::Dual{_InnerTag}, i::Int) = partials(y, i)
@inline _ip(y, ::Int) = zero(y)

# GPU-safe gradient/jacobian/derivative: seed `Dual{_InnerTag}` and extract
# `partials` directly, bypassing `extract_jacobian`/`valtype`. SVector inputs
# take the GPU-safe path; other inputs (plain `AbstractVector`, e.g. unit tests)
# dispatch to ForwardDiff (CPU-only)
@inline function gradient(f::F, x::SVector{N,T}) where {F,N,T}
    seeds = ntuple(i -> Dual{_InnerTag}(x[i], ntuple(j -> ifelse(j==i, one(T), zero(T)), Val(N))), Val(N))
    y = f(SVector(seeds))
    SVector(ntuple(j -> _ip(y, j), Val(N)))
end
@inline function jacobian(f::F, x::SVector{N,T}) where {F,N,T}
    seeds = ntuple(i -> Dual{_InnerTag}(x[i], ntuple(j -> ifelse(j==i, one(T), zero(T)), Val(N))), Val(N))
    _stack_jac(f(SVector(seeds)), Val(N))
end
@inline function _stack_jac(ydual::SVector{M}, ::Val{N}) where {M,N}
    SMatrix{M,N}(ntuple(k -> _ip(ydual[((k-1) % M) + 1], ((k-1) ÷ M) + 1), Val(M*N)))
end
@inline derivative(f::F, t::T) where {F,T} = map(yi -> _ip(yi, 1), f(Dual{_InnerTag}(t, one(T))))
@inline gradient(f, x) = ForwardDiff.gradient(f, x)
@inline jacobian(f, x) = ForwardDiff.jacobian(f, x)