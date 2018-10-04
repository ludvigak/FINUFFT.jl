# FINUFFT.jl

[![Build Status](https://travis-ci.org/ludvigak/FINUFFT.jl.svg?branch=master)](https://travis-ci.org/ludvigak/FINUFFT.jl)

This is a Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), a lightweight and fast nonuniform FFT (nufft) library released by the Flatiron Institute.

## Installation

Julia 1.0 and 0.7: From the Pkg REPL mode (hit `]` in REPL to enter), run
```julia
add FINUFFT
test FINUFFT
```

Julia 0.6: From the REPL, run
```julia
Pkg.add("FINUFFT")
Pkg.test("FINUFFT")
```

This should download, build and test FINUFFT v1.0, as long as you have `gcc` and `curl` installed. The FFTW library is downloaded locally by the build script, using [Conda.jl](https://github.com/JuliaPy/Conda.jl) 

Currently only tested on Linux.

## Usage

This module provides functions `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!` and `finufft_default_opts` that call the interface defined in <https://finufft.readthedocs.io/en/latest/usage.html>

* Function calls mimic the C/C++ interface, with the exception that you don't need to pass any array sizes (they are inferred from the sizes of the arguments).
* Output arrays need to be preallocated and passed as arguments.
* The last argument of the nufft routines is the options struct, which is optional. Default values are used if it is omitted.
* `finufft_default_opts()` returns an options struct with default values.

Example:
```julia
using FINUFFT

# nonuniform data
nj = 100
x = pi*(1.0 .- 2.0*rand(nj))
c = rand(nj) + 1im*rand(nj)

# Allocate output
ms = 20
out = Array{ComplexF64}(undef, ms)

# Call FINUFFT using default opts
tol = 1e-10
nufft1d1!(x, c, 1.0, tol, out)

# Call FINUFFT using modified opts 
opts = finufft_default_opts()
opts.debug = 1
nufft1d1!(x, c, 1, tol, out, opts)
```

The advanced interfaces `finufft2d1many` and `finufft2d2many` have not been implemented yet.

### More examples
See [test/runtests.jl](test/runtests.jl)

## TODO
* Implement advanced interface
* Test Max OS X, once it is available in FINUFFT 

