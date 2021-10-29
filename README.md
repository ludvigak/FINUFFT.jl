# FINUFFT.jl

[![CI](https://github.com/ludvigak/FINUFFT.jl/workflows/CI/badge.svg?branch=master)](https://github.com/ludvigak/FINUFFT.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/ludvigak/FINUFFT.jl/branch/master/graph/badge.svg?token=Tkx7kma18J)](https://codecov.io/gh/ludvigak/FINUFFT.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ludvigak.github.io/FINUFFT.jl/latest/)

This is a full-featured Julia interface to [FINUFFT](https://github.com/flatironinstitute/finufft), a lightweight and fast nonuniform FFT (nufft) library released by the Flatiron Institute.

## Installation

FINUFFT.jl requires Julia v1.3 or later. From the Pkg REPL mode (hit `]` in REPL to enter), run

```julia
add FINUFFT
```

This installs dependencies, including the generic multi-platform precompiled
binaries [finufft_jll.jl](https://github.com/JuliaBinaryWrappers/finufft_jll.jl). You may get more performance by locally compiling.

Older versions of the package are available also for Julia v1.0-v1.2, but the user need to have a recent version of GCC installed.

## Usage

This module now provides the functions `nufft1d1`, `nufft1d2`, ..., `nufft3d3`, `nufft1d1!`, `nufft1d2!`, ..., `nufft3d3!`, that now wrap the
simple and vectorized interfaces in a unified way,
as well as
`finufft_makeplan`, `finufft_setpts`, `finufft_exec`, `finufft_exec!` and `finufft_destroy` that wrap the guru interface.
This brings the Julia interface up to the standards of the
FINUFFT's [MATLAB/Octave](https://finufft.readthedocs.io/en/latest/matlab.html)
and [Python](https://finufft.readthedocs.io/en/latest/python.html) interfaces.
The underlying C++ routines that are called have full documentation
[here](https://finufft.readthedocs.io/en/latest/c.html).

An auto-generated reference for all provided Julia functions is [here](https://ludvigak.github.io/FINUFFT.jl/latest/).

.. note::

   As of 10/28/21, the interface has changed (improved) significantly,
   breaking backward-compatibility.

* Function calls mimic the C/C++ interface, with the exception that you don't need to pass the dimensions of any arrays in the argument (they are inferred using `size()`).
* A vectorized call (performing multiple transforms, each with different coefficient vectors but the same set of nonuniform points) can now be performed using the same functions as the single-transform interface, detected from the size of the input arrays.
* Both 64-bit and 32-bit precision calls are now supported using a single
set of function names, switched by a `dtype` keyword argument for clarity.
* The functions named `nufftDdN` return the output array.
* In contrast, the functions named `nufftDdN!` take the output array as an argument. This needs to be preallocated with the correct size.
* Likewise, in the guru interface, `finufft_exec` returns the output array,
while `finufft_exec!` takes the output array as an argument that needs to be preallocated.
* Options differing from their default values are now set using keyword arguments both in the simple interfaces, or in `finufft_makeplan` for the guru interface.

### Example
```julia
using FINUFFT

# nonuniform data
nj = 100
x = pi*(1.0 .- 2.0*rand(nj))
c = rand(nj) + 1im*rand(nj)

# Parameters
ms = 20      # Output size
tol = 1e-10  # Tolerance

# Output as return value
fk = nufft1d1(x, c, 1, tol, ms)

# Or, preallocate output and pass as argument
out = Array{ComplexF64}(undef, ms)
nufft1d1!(x, c, 1, tol, out)

# Demo using kwargs to change options from defaults...
fk2 = nufft1d1(x, c, 1, tol, ms, debug=1, dtype=Float64)
```

### More examples

See the code [test/test_nufft.jl](test/test_nufft.jl)

### Developers

Main authors:

* Ludvig af Klinteberg (old interface)
* Libin Lu (new full-featured interface)
* Jonas Krimmer (many contributions to full-featured interface)

Additional authors:

* Alex Barnett (guidance/tweaks)
* Mose Giordano (packaging)
