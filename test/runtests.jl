using WaterLily
import Pkg, ParallelTestRunner

#=
Parallel test runner (ParallelTestRunner.jl): every test_*.jl in this directory runs in
its own isolated module, in parallel across worker processes. Each worker gets
WATERLILY_NTHREADS threads (default 2, to exercise the multithreaded KernelAbstractions
CPU path); --jobs is auto-sized to ≈ Sys.CPU_THREADS / WATERLILY_NTHREADS unless given.

Test suite set by the WaterLily `backend` preference (LocalPreferences.toml):
    "KernelAbstractions" -> main test sets (every test_*.jl except test_alloc.jl)
    "SIMD"               -> allocation tests (test_alloc.jl) only

Array backends: WATERLILY_BACKENDS=cpu|cuda|amdgpu|metal|all (comma-separated, default "cpu").
GPU backends are opt-in and installed on demand (they are not test dependencies).
Metal has no Float64 support, so only Float32 test sets (e.g. "float32") can pass on it.
So "all" does not include it: request "metal" explicitly (Apple only).

Reverse-mode AD with Enzyme (test_enzyme.jl) is opt-in as well: WATERLILY_ENZYME=true.
Packages installed on demand are only added when they are not loadable already, so running
this file directly in an environment with developed checkouts keeps them, e.g.
    WATERLILY_ENZYME=true julia --project=<env> test/runtests.jl enzyme

Run a subset by passing set name(s) (matched with startswith) and/or runner flags
(--jobs=N, --list, --verbose, --quickfail), e.g.
    julia --project -e 'using Pkg; Pkg.test(test_args=["poisson","flow"])'
=#

const WATERLILY_BACKENDS = filter(!isempty, strip.(split(lowercase(get(ENV, "WATERLILY_BACKENDS", "cpu")), ',')))
(isempty(WATERLILY_BACKENDS) || any(b -> b ∉ ("cpu","cuda","amdgpu","metal","all"), WATERLILY_BACKENDS)) &&
    throw(ArgumentError("WATERLILY_BACKENDS must be a comma-separated list of cpu|cuda|amdgpu|metal|all, got \"$(get(ENV, "WATERLILY_BACKENDS", ""))\""))
_cpu = any(b -> b in ("cpu","all"), WATERLILY_BACKENDS)
_cuda = any(b -> b in ("cuda","all"), WATERLILY_BACKENDS)
_amdgpu = any(b -> b in ("amdgpu","all"), WATERLILY_BACKENDS)
_metal = "metal" in WATERLILY_BACKENDS && Sys.isapple() # not part of "all" since most test sets are Float64

_enzyme = lowercase(get(ENV, "WATERLILY_ENZYME", "false")) == "true"

# GPU packages and Enzyme are not test deps: install the requested ones once here in the main process,
# unless they are loadable already. A package that cannot be installed is skipped
function ensure(pkg)
    isnothing(Base.find_package(pkg)) || return true
    try Pkg.add(pkg); return true catch e; @warn "Requested $pkg could not be installed; skipping" exception=e; return false end
end
_cuda = _cuda && ensure("CUDA")
_amdgpu = _amdgpu && ensure("AMDGPU")
_metal = _metal && ensure("Metal")
_enzyme = _enzyme && ensure("Enzyme")

# Discover test sets: every test_*.jl, gated by the WaterLily `backend` preference
const TESTDIR = @__DIR__
is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")
setname(f) = f[6:end-3] # "test_core.jl" -> "core"
wanted(name) = name == "enzyme" ? _enzyme : backend == "SIMD" ? name == "alloc" : name != "alloc"
testsuite = Dict{String,Expr}()
for (root, _dirs, files) in walkdir(TESTDIR), f in filter(is_test_file, files)
    name = setname(f)
    wanted(name) && (testsuite[name] = :(include($(joinpath(root, f)))))
end

# Per-sandbox setup (runs in each isolated module): imports, the `arrays` list and helpers
const init_code = quote
    using WaterLily, Test, StaticArrays
    import GPUArrays
    arrays = []
    $(_cpu)    && push!(arrays, Array)
    $(_cuda)   && using CUDA
    $(_amdgpu) && using AMDGPU
    $(_metal)  && using Metal
    $(_cuda)   && CUDA.functional()   && push!(arrays, CUDA.CuArray)
    $(_amdgpu) && AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)
    $(_metal)  && Metal.functional()  && push!(arrays, Metal.MtlArray)
    isempty(arrays) && error("No functional backend available")
    include($(joinpath(TESTDIR, "helper.jl")))
end

# WATERLILY_NTHREADS (default 2) gives every worker that many threads,
# overriding ParallelTestRunner pinning to single thread, so that tests
# run with KA multi-threading.
const _nt = get(ENV, "WATERLILY_NTHREADS", "2")
const WATERLILY_NTHREADS = isempty(_nt) ? 2 : parse(Int, _nt)
exeflags = WATERLILY_NTHREADS == 1 ? nothing : ["--threads=$WATERLILY_NTHREADS"]

# Auto-size --jobs so jobs × WATERLILY_NTHREADS ≈ the machine's thread count, unless the
# caller passed --jobs explicitly (e.g. via Pkg.test(test_args=["--jobs=N"])).
args = copy(ARGS)
any(a -> startswith(a, "--jobs"), args) ||
    push!(args, "--jobs=$(max(1, Sys.CPU_THREADS ÷ WATERLILY_NTHREADS))")

ParallelTestRunner.runtests(WaterLily, args; testsuite, init_code, exeflags)
