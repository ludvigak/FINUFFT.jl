# FINUFFT.jl

This is a Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), a lightweight nonuniform FFT (nufft) library released by the Flatiron Institute.

## Usage

This module provides functions `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!` and `finufft_default_opts` that call the C interface defined in <https://finufft.readthedocs.io/en/latest/usage.html#id3>

### Examples
See [test/runtests.jl](test/runtests.jl)
