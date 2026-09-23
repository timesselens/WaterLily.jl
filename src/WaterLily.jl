"""
$(README)
"""
module WaterLily

using DocStringExtensions, Printf

abstract type AbstractSimulation end

include("core.jl")
export BC!,@inside,inside,δ,loc,@log,set_backend,backend

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!,L₂

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult!

include("Flow.jl")
export AbstractFlow,Flow,mom_step!,quick,cds,apply!

include("Body.jl")
export AbstractBody,measure_sdf!

include("AutoBody.jl")
export AutoBody,Bodies,measure,sdf,+,-

include("Metrics.jl")
export MeanFlow,update!,uu!,uu

include("util.jl")

include("RigidMap.jl")
export RigidMap,setmap

"""
    check_fn(f,N,T,nargs)

Check a user-supplied BC/IC function `f(i,x)` or `f(i,x,t)` has `nargs`
arguments and is type-stable, returning `T` for every `i in 1:N`. No-op if
`f` is not a `Function`.
"""
check_fn(f,N,T,nargs) = nothing
function check_fn(f::Function,N,T,nargs)
    @assert first(methods(f)).nargs==nargs+1 "$f signature needs $nargs arguments"
    @assert all(typeof.(ntuple(i->f(i,xtargs(Val{}(nargs),N,T)...),N)).==T) "$f is not type stable"
end
xtargs(::Val{2},N,T) = (zeros(SVector{N,T}),)
xtargs(::Val{3},N,T) = (zeros(SVector{N,T}),zero(T))
"""
    Simulation(dims::NTuple, uBC::Union{NTuple,Function}, L::Number;
               U=nothing, Δt=0.25, ν=0., ϵ=1, g=nothing,
               u0=nothing, perdir=(), exitBC=false, λ=quick,
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array,
               flow_ctor=(dims,uBC;kw...)->Flow(dims,uBC;kw...),
               pois_ctor=flow->MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `uBC`: Velocity field applied to boundary and acceleration conditions.
        Define a `Tuple` for constant BCs, or a `Function` for space and time varying BCs `uBC(i,x,t)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale. Required if using `uBC::Function`.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `g`: Domain acceleration, `g(i,x,t)=duᵢ/dt`
  - `ϵ`: BDIM kernel width.
  - `perdir`: Domain periodic boundary condition in the `(i,)` direction.
  - `λ`: Convective interpolation scheme `λ(u,c,d)` (upstream, central, downstream), e.g. `quick` (default),
        `cds`, or `vanLeer`.
  - `u0`: Velocity field applied to the initial condition.
        Define a Tuple for homogeneous (per direction) IC, or a `Function` for space varying IC `u0(i,x)`.
        The deprecated `uλ` keyword is still accepted as an alias and will be removed in a future release.
  - `exitBC`: Convective exit boundary condition in the `i=1` direction.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.
  - `flow_ctor`: Factory callable `(dims, uBC; kw...) -> AbstractFlow` to substitute a custom flow type.
        The callable receives all standard keyword arguments forwarded from this constructor.
        Used by downstream packages (e.g. LilyPad.jl) to inject a custom `AbstractFlow` subtype.
  - `pois_ctor`: Factory callable `flow -> AbstractPoisson` to substitute a custom Poisson solver.
        Called after `flow_ctor` with the constructed flow as argument.
        Used by downstream packages (e.g. BiotSavartBCs.jl) to inject a custom `AbstractPoisson` subtype.

See the repository [WaterLily-Examples](https://github.com/WaterLily-jl/WaterLily-Examples) for more examples.
"""
mutable struct Simulation{F<:AbstractFlow,P<:AbstractPoisson} <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: F # concretely typed: `sim.flow` and `sim.pois` are type-stable (and never replaced)
    body :: AbstractBody
    pois :: P
    function Simulation(dims::NTuple{N}, uBC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(),
                        u0=nothing, uλ=nothing, exitBC=false, λ=quick, body::AbstractBody=NoBody(),
                        flow_ctor=(dims,uBC;kw...)->Flow(dims,uBC;kw...),
                        pois_ctor=flow->MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir),
                        T=Float32, mem=Array) where N
        @assert !(isnothing(U) && isa(uBC,Function)) "`U` (velocity scale) must be specified if boundary conditions `uBC` is a `Function`"
        isnothing(U) && (U = √sum(abs2,uBC))
        u0 = ic_kwarg(u0, uλ)
        check_fn(uBC,N,T,3); check_fn(g,N,T,3); check_fn(u0,N,T,2)
        flow = flow_ctor(dims,uBC;u0,Δt,ν,g,λ,T,mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        pois = pois_ctor(flow)
        new{typeof(flow),typeof(pois)}(U,L,ϵ,flow,body,pois)
    end
end

time(sim::AbstractSimulation) = time(sim.flow)
"""
    sim_time(sim::AbstractSimulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::AbstractSimulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::AbstractSimulation,t_end;remeasure=true,max_steps=typemax(Int),verbose=false,
        udf=nothing,poisson_kwargs=(;),kwargs...)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step. Can be set to `false` for static geometries to speed up simulation.
A user-defined function `udf` can be passed to arbitrarily modify the `::AbstractFlow` during the predictor and corrector steps.
If the `udf` user keyword arguments, these needs to be included in the `sim_step!` call as well.
`poisson_kwargs` (eg. `(;tol=1e-4,itmx=10)`) controls the pressure `solver!` and is kept
separate from the `udf` keyword arguments, so it is never forwarded into `udf`.
"""
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,max_steps=typemax(Int),verbose=false,
        udf=nothing,poisson_kwargs=(;),kwargs...)
    evolve!(sim,t_end;max_steps,verbose) do sim
        sim_step!(sim; remeasure, udf, poisson_kwargs, kwargs...)
    end
    return nothing
end
"""
    evolve!(step!,sim::AbstractSimulation,t_end;max_steps=typemax(Int),verbose=false,init=nothing)

Call `step!(sim)` until the dimensionless time `t_end` (or `max_steps` steps), and return `init`
plus the sum of what `step!` returns (values of `nothing` are skipped). `sim_step!(sim,t_end)` is
`evolve!` with a plain time step; a running cost, e.g. a time-averaged force, is accumulated by
returning it from `step!` (with `init` a zero of its type). With Enzyme, reverse mode through
`evolve!` stores one checkpoint per step instead of the whole trajectory, and replays the steps
backwards, which leaves the flow as it was before `evolve!`. The checkpoint holds `u`, `p`, `σ` and
the time steps, so `step!` must recompute everything else it reads (as `sim_step!` does).
"""
function evolve!(step!::F,sim::AbstractSimulation,t_end;max_steps=typemax(Int),verbose=false,init=nothing) where F
    steps₀ = length(sim.flow.Δt); J = init
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        J = accumulate_cost(J,step!(sim))
        verbose && sim_info(sim)
    end
    return J
end
accumulate_cost(J,::Nothing) = J
accumulate_cost(::Nothing,c) = c
accumulate_cost(::Nothing,::Nothing) = nothing
accumulate_cost(J,c) = J+c
function sim_step!(sim::AbstractSimulation;remeasure=true,udf=nothing,poisson_kwargs=(;),kwargs...)
    remeasure && measure!(sim)
    mom_step!(sim.flow, sim.pois; udf, poisson_kwargs, kwargs...)
    return nothing
end

"""
    measure!(sim::AbstractSimulation,t=sum(sim.flow.Δt))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::AbstractSimulation,t=sum(sim.flow.Δt))
    with_body(measure_body!,sim,t,sim.ϵ)
    update!(sim.pois)
end
measure_body!(flow,body,t,ϵ) = measure!(flow,body;t,ϵ)

# `sim.body` is abstractly typed (a body can be replaced), so a call that takes it is dispatched at run time.
# `with_body(f,sim,args...)` calls `f(sim.flow,sim.body,args...)` behind a function barrier: the run-time
# dispatch only sees `sim`, the body's type and `args` (for `measure!`, `ϵ`, which is abstractly typed too),
# and `f` reads the body with its type known. Enzyme needs this: it passes an immutable body with active
# parameters to a call that is dispatched at run time only through a `Ref`, and the shadow that
# `Enzyme.make_zero(sim)` builds holds the body itself.
with_body(f::F,sim,args...) where F = _with_body(f,sim,typeof(sim.body),args...)
_with_body(f::F,sim,::Type{B},args...) where {F,B} = f(sim.flow,sim.body::B,args...)

"""
    sim_info(sim::AbstractSimulation)
Prints information on the current state of a simulation.
"""
sim_info(sim::AbstractSimulation) = @printf "tU/L=%.4f, Δt=%.3f\n" sim_time(sim) sim.flow.Δt[end]

"""
    perturb!(sim::AbstractSimulations; noise=0.1)
Perturb the velocity field of a simulation with `noise` level with respect to velocity scale `U`.
"""
perturb!(sim::AbstractSimulation; noise=0.1) = sim.flow.u .+= randn(size(sim.flow.u))*sim.U*noise |> typeof(sim.flow.u).name.wrapper

export AbstractSimulation,Simulation,sim_step!,evolve!,sim_time,measure!,sim_info,perturb!

# defaults JLD2 and VTK I/O functions
function load!(sim::AbstractSimulation; kwargs...)
    fname = get(Dict(kwargs), :fname, "WaterLily.jld2")
    ext = split(fname, ".")[end] |> Symbol
    vtk_loaded = !isnothing(Base.get_extension(WaterLily, :WaterLilyReadVTKExt))
    jld2_loaded = !isnothing(Base.get_extension(WaterLily, :WaterLilyJLD2Ext))
    ext == :pvd && (@assert vtk_loaded "WriteVTK must be loaded to save .pvd data.")
    ext == :jdl2 && (@assert jld2_loaded "JLD2 must be loaded to save .jld2 data.")
    load!(sim, Val{ext}(); kwargs...)
end
function save! end
function vtkWriter end
function default_attrib end
function pvd_collection end
export load!, save!, vtkWriter, default_attrib

# default Plots functions
function flood end
function addbody end
function body_plot! end
function sim_gif! end
function plot_logger end
# export
export flood,addbody,body_plot!,sim_gif!,plot_logger

# default Makie functions
function viz! end
function viz_step! end
function get_body end
function plot_body_obs! end
# hook set by WaterLilyPathlinesExt.__init__ when Pathlines is loaded
const _pathlines_viz_hook = Ref{Union{Nothing,Function}}(nothing)
# export
export viz!, viz_step!, get_body, plot_body_obs!

# Check number of threads when loading WaterLily
"""
    check_nthreads()

Check the number of threads available for the Julia session that loads WaterLily.
Called from `__init__`; skipped during package precompilation (which always runs with one thread).
A warning is shown when running in serial (JULIA_NUM_THREADS=1) with KernelAbstractions enabled.
"""
function check_nthreads()
    Base.generating_output() && return nothing
    if backend == "KernelAbstractions" && Threads.nthreads() == 1
        @warn """
        Using WaterLily in serial (ie. JULIA_NUM_THREADS=1) is not recommended because it defaults to serial CPU execution.
        Use JULIA_NUM_THREADS=auto, or any number of threads greater than 1, to allow multi-threading in CPU backends.
        For a low-overhead single-threaded CPU only backend set: WaterLily.set_backend("SIMD")
        """
    end
end

__init__() = check_nthreads()
    
end # module
