# FINUFFT.jl

[![CI](https://github.com/ludvigak/FINUFFT.jl/workflows/CI/badge.svg?branch=master)](https://github.com/ludvigak/FINUFFT.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/ludvigak/FINUFFT.jl/branch/master/graph/badge.svg?token=Tkx7kma18J)](https://codecov.io/gh/ludvigak/FINUFFT.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ludvigak.github.io/FINUFFT.jl/latest/)

This is a Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), a lightweight and fast nonuniform FFT (nufft) library released by the Flatiron Institute.
Both 64-bit and 32-bit precisions calls are supported.


## Installation

The latest version of FINUFFT.jl requires Julia v1.3 or later. From the Pkg REPL mode (hit `]` in REPL to enter), run

```julia
add FINUFFT
```

Older versions of the package are available also for Julia v1.0-v1.2, but the user need to have a recent version of GCC installed.

## Usage

This module provides functions `nufft1d1`, `nufft1d2`, ..., `nufft3d3`, `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!`, `finufft_makeplan`, `finufft_setpts`, `finufft_exec`, `finufft_exec!` and `finufft_destroy` that call the interface defined in <https://finufft.readthedocs.io/en/latest/usage.html>

A reference of the provided functions is available at <https://ludvigak.github.io/FINUFFT.jl/latest/>

* Function calls mimic the C/C++ interface, with the exception that you don't need to pass the dimensions of any arrays in the argument (they are inferred using `size()`).
* The functions named `nufftDdN` return the output array.
* The functions named `nufftDdN!` take the output array as an argument. This needs to be preallocated.
* `finufft_exec` returns the output array.
* `finufft_exec!` takes the output array as an argument. This needs to be preallocated.
* The keyword arguments of the nufft routines are used to set up the options struct. Default values are used if they are omitted.

### Example
```julia
using FINUFFT

# nonuniform data
nj = 100
x = pi*(1.0 .- 2.0*rand(nj))
c = rand(nj) + 1im*rand(nj)

# Parameters
ms = 20 # Output size
tol = 1e-10 # Tolerance

# Output as return value
fk = nufft1d1(x, c, 1, tol, ms)

# Preallocate output and pass as argument
out = Array{ComplexF64}(undef, ms)
nufft1d1!(x, c, 1, tol, out)

# Call using kwargs
fk2 = nufft1d1(x, c, 1, tol, ms, debug=1, spread_kerpad=0, dtype=Float64)
```

### More examples
See [test/test_nufft.jl](test/test_nufft.jl)
