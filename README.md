# FINUFFT.jl

[![Build Status](https://travis-ci.org/ludvigak/FINUFFT.jl.svg?branch=master)](https://travis-ci.org/ludvigak/FINUFFT.jl)
[![Coverage Status](https://coveralls.io/repos/github/ludvigak/FINUFFT.jl/badge.svg?branch=master)](https://coveralls.io/github/ludvigak/FINUFFT.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ludvigak.github.io/FINUFFT.jl/latest/)

This is a Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), a lightweight and fast nonuniform FFT (nufft) library released by the Flatiron Institute.

## Installation

Julia 1.0 and up: From the Pkg REPL mode (hit `]` in REPL to enter), run
```julia
add FINUFFT
test FINUFFT
```

This should download, build and test FINUFFT v1.1.1, as long as you have `gcc` and `curl` installed. The FFTW library is downloaded locally by the build script, using [Conda.jl](https://github.com/JuliaPy/Conda.jl) 

Developed and tested on Linux. Also works on Max OS X, but build script is hardwired to use GCC 8 (`g++-8` and `gcc-8`).

Might work on Windows as well. Has been tested successfully on two Windows 10 setups with TDM-GCC 9.2.0 and Julia 1.4.1. (Important: make sure that openmp for gcc is installed.). 

## Usage

This module provides functions `nufft1d1`, `nufft1d2`, ..., `nufft3d3`, `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!`, and `finufft_default_opts` that call the interface defined in <https://finufft.readthedocs.io/en/latest/usage.html>

A reference of the provided functions is available at <https://ludvigak.github.io/FINUFFT.jl/latest/>

* Function calls mimic the C/C++ interface, with the exception that you don't need to pass the dimensions of any arrays in the argument (they are inferred using `size()`).
* The functions named `nufftDdN` return the output array.
* The functions named `nufftDdN!` take the output array as an argument. This needs to be preallocated.
* The last argument of the nufft routines is the options struct, which is optional. Default values are used if it is omitted.
* `finufft_default_opts()` returns an options struct with default values.
* The advanced interfaces `finufft2d1many` and `finufft2d2many` have not been implemented yet.

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

# Call using modified opts 
opts = finufft_default_opts()
opts.debug = 1
fk2 = nufft1d1(x, c, 1, tol, ms, opts)
```

### More examples
See [test/test_nufft.jl](test/test_nufft.jl)

## TODO
* Implement advanced interface
