# FINUFFT.jl

[![CI](https://github.com/ludvigak/FINUFFT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ludvigak/FINUFFT.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/ludvigak/FINUFFT.jl/branch/master/graph/badge.svg?token=Tkx7kma18J)](https://codecov.io/gh/ludvigak/FINUFFT.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ludvigak.github.io/FINUFFT.jl/latest/)

This is a full-featured Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), which is a lightweight and fast parallel nonuniform fast Fourier transform (NUFFT) library released by the Flatiron Institute, and its GPU version cuFINUFFT. This interface stands at v3.x, and it uses FINUFFT version 2.4.0-rc1 (note that the interface version number is distinct from the version of the wrapped binary FINUFFT library).

## Installation

FINUFFT.jl requires Julia v1.6 or later, and has been tested up to v1.11. From the Pkg REPL mode (hit `]` in REPL to enter), run

```julia
add FINUFFT
```

This installs the stable registered version and its dependencies, including our multi-platform precompiled binaries of the FINUFFT library:

* [finufft_jll.jl](https://github.com/JuliaBinaryWrappers/finufft_jll.jl), which contains the CPU routines and has microarchitecture-specific builds (including `avx512`) for better performance.

* [cufinufft_jll.jl](https://github.com/JuliaBinaryWrappers/cufinufft_jll.jl), which contains the GPU routines (cuFINUFFT). This is only available for x86 GNU Linux, and is built for CUDA versions up to 12.6.

To access the GPU interface, CUDA.jl must be installed through `add CUDA`

You may be able to squeeze a little more performance via locally-compiled binaries; see below.

## Usage

This module provides multiple sets of functions:

* `nufft1d1`, `nufft1d2`, ..., `nufft3d3` wrap the
simple and vectorized interfaces in a unified way,
* `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!` work with pre-allocated arrays, and also interface to the GPU routines.
* `finufft_makeplan`, `finufft_setpts!`, `finufft_exec`, `finufft_exec!`, `finufft_destroy!` directly wrap the [guru interface](https://finufft.readthedocs.io/en/latest/c.html#guru-plan-interface).
* `cufinufft_makeplan`, `cufinufft_setpts!`, `cufinufft_exec`, `cufinufft_exec!`, `cufinufft_destroy!` directly wrap the [cuFINUFFT interface](https://finufft.readthedocs.io/en/latest/c_gpu.html).

The underlying C++ routines that are called have full documentation
[here](https://finufft.readthedocs.io/en/latest/c.html).

An auto-generated reference for all provided Julia functions is [here](https://ludvigak.github.io/FINUFFT.jl/latest/).

* Function calls mimic the C/C++ interface, with the exception that you don't need to pass the dimensions of any arrays in the argument (they are inferred using `size()`).
* A vectorized call (performing multiple transforms, each with different coefficient vectors but the same set of nonuniform points) can now be performed using the same functions as the single-transform interface, detected from the size of the input arrays.
* Both 64-bit and 32-bit precision calls are supported using a single
set of function names. Which precision to use is inferred from the type of the input arrays, except for in the guru interface where the `dtype` argument is required for `finufft_makeplan`. (NOTE: The use of the `dtype` argument in the simple interface is deprecated as of v3.1.0)
* The functions named `nufftDdN` return the output array.
* In contrast, the functions named `nufftDdN!` take the output array as an argument. This needs to be preallocated with the correct size.
* `nufftDdN!` infers from the array types whether it should run on CPU or GPU.
* Likewise, in the guru/GPU interfaces, `[cu]finufft_exec` returns the output array,
while `[cu]finufft_exec!` takes the output array as an argument that needs to be preallocated. The methods `[cu]finufft_setpts!` and `[cu]finufft_destroy!` include explamation points in Julian style, since they both change the plan.
* Options differing from their default values are set using keyword arguments both in the simple interfaces, or in `[cu]finufft_makeplan` for the guru/GPU interfaces.
* The GPU interface is only loaded into FINUFFT.jl if CUDA.jl is loaded into the same environment, i.e. `using FINUFFT, CUDA`. This is to reduce load times for non-GPU users, as CUDA.jl takes several seconds to load.

### Example
```julia
using FINUFFT

# Here we demo a Float64 1D type 1 transform
nj = 1000000
x = pi*(1.0 .- 2.0*rand(nj))      # nonuniform points
c = rand(nj) + 1im*rand(nj)       # their strengths

ms = 2000000      # output size (number of Fourier modes)
tol = 1e-9        # requested relative tolerance

# Output as return value (1e6 pts to 2e6 modes takes about 0.1 sec)...
fk = nufft1d1(x, c, 1, tol, ms)

# Or, output into preallocated array, whose size determines ms...
out = Array{ComplexF64}(undef, ms)
nufft1d1!(x, c, 1, tol, out)

# Demo using keyword args to change options from defaults...
fk_fftord = nufft1d1(x, c, 1, tol, ms, debug=1, modeord=1, nthreads=4)
```

The above code may be found in [examples/demo1d1.jl](examples/demo1d1.jl)

### GPU Example
Continue the above code with:
```julia
using CUDA # CUDA must be loaded for cuFINUFFT to be activated

# Copy input data to GPU, "_d" suffix indiciates data on device (GPU)
x_d = CuArray(x)
c_d = CuArray(c)

# Allocate CUDA aray
out_d = CuArray{ComplexF64}(undef, ms)

# Note: identical interface as CPU, but with CUDA arrays on device
nufft1d1!(x_d, c_d, 1, tol, out_d)

# Copy results back to host memory
gpu_results = Array(out_d)

```

The above code may be found in [examples/demo1d1_cuda.jl](examples/demo1d1_cuda.jl)

### More examples

For a 2D type 1 with timing benchmark,
see [examples/time2d1.jl](examples/time2d1.jl)

Finally, the more involved codes [test/test_nufft.jl](test/test_nufft.jl) 
and [test/test_cuda.jl](test/test_cuda.jl)
test `dtype=Float64` and `dtype=Float32` precisions for all supported transform types, and can be used as references.
The outputs are tested there for mathematical correctness.
In the 1D type 1 it also tests a vectorized simple, a guru call and
a vectorized guru call.
The help documentation for each function will also gradually be populated
with examples.



## Advanced installation and locally compiling binaries

To get the latest version of this interface do `add FINUFFT#master`, but note this still uses the precompiled binaries from `[cu]finufft_jll`.

You may get a little more performance by locally compiling binaries as follows. This has only been tested on ubuntu linux, so YMMV. First install the source
[FINUFFT](https://github.com/flatironinstitute/finufft) and build it according to its documentation.
Now start Julia and install the latest interface in develop mode:
```julia
pkg> dev https://github.com/ludvigak/FINUFFT.jl
```
This should create `~/.julia/dev/FINUFFT/src/FINUFFT.jl` which you should edit,
following the simple instructions to set
```julia
const libfinufft = "YOURFINUFFT/lib/libfinufft.so"
```
Similarly, in `~/.julia/dev/FINUFFT/src/cufinufft_jll.jl` after the line `using cufinufft_jll` add
```julia
cufinufft_jll.libcufinufft = "YOURFINUFFT/lib/libcufinufft.so"

```

Restart Julia, and `pkg> test FINUFFT` to check it worked.
You may find that `julia> include("examples/time2d1.jl")` runs faster
than before (however, since we included `avx2` in our binaries, it is
unlikely to run faster on an x86_64 CPU).
Now proceed by `using FINUFFT` as usual.
You may do `pkg> free FINUFFT` and restart to return to the registered package
with generic binaries.
Here's [general info about packages](https://pkgdocs.julialang.org/v1/managing-packages).

Finally, older versions of the package are available also for Julia v1.0-v1.2, but the user needs to have a recent version of GCC installed.


## Developers of this Julia wrapper

Main authors:

* Ludvig af Klinteberg (maintainer, old interface, binaries, CUDA interface)
* Libin Lu (new full-featured interface)
* Jonas Krimmer (many contributions to full-featured interface)

Additional authors:

* Alex Barnett (guidance/tweaks/docs/examples)
* Mose Giordano (packaging, binaries)

### To do (please help)

- populate the docstrings each with a working example
- add more `examples/` with math tests
- more extensive tests, including more "dumb inputs" as in C++
