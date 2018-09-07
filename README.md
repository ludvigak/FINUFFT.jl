# FINUFFT.jl

This is a Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), a lightweight nonuniform FFT (nufft) library released by the Flatiron Institute.

## Installation

Install through your Julia package manager. This should download and build FINUFFT v1.0, as long as you satisfy the dependencies listed in <https://finufft.readthedocs.io/en/latest/install.html#dependencies>

## Usage

This module provides functions `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!` and `finufft_default_opts` that call the interface defined in <https://finufft.readthedocs.io/en/latest/usage.html>

* Function calls mimic the C/C++ interface, with the exception that you don't need to pass any array sizes.
* Output arrays need to be preallocated and passed as arguments.

```julia
using FINUFFT
```

The advanced interfaces `finufft2d1many` and `finufft2d2many` have not been implemented yet.

### Examples
See [test/runtests.jl](test/runtests.jl)
