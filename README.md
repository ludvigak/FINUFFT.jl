# FINUFFT.jl

This is a Julia wrapper for [FINUFFT](https://github.com/ahbarnett/finufft), a lightweight nonuniform FFT (nufft) library written by Alex H. Barnett and Jeremy F. Magland.

## Installation

This is just a secondary wrapper around FINUFFT's Python wrapper, so first you need to build the library and its Python wrappers by following the instructions here: <http://finufft.readthedocs.io/en/latest/install.html>

Then install this wrapper with the Julia command
```julia
Pkg.clone("https://github.com/ludvigak/FINUFFT.jl.git")
```

## Usage

This module provides functions `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!` that wrap the interfaces defined in <http://finufft.readthedocs.io/en/latest/pythoninterface.html>

The only difference between using this module and writing
```julia
using PyCall
@pyimport finufftpy
```
is that you with this module get function names that end with a `!`, which is the convention for functions that modify their arguments.

### Examples
See [test/runtests.jl](test/runtests.jl)
