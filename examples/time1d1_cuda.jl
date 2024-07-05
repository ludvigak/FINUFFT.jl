## Here we demo CUDA routines using the 1D type 1 transform
# single-prec speed comparison.

using FINUFFT
using LinearAlgebra
using BenchmarkTools

dtype = Float32 # Datatype for computations
tol   = 1e-5   # requested relative tolerance

# Setup problem
nj = Int(3e8)
x = pi*(1 .- 2*rand(dtype, nj)); # nonuniform points
c = rand(Complex{dtype}, nj);    # their strengths
ms = Int(1e6)                    # output size (number of Fourier modes)

# CPU computation with preallocated array
fk = Array{Complex{dtype}}(undef, ms);

@btime nufft1d1!(x, c, 1, tol, fk)       # 6 sec on 10 threads of xeon 8358
                                        # (0.16G NUpt/s)

##############################################
## Simple GPU interface for preallocated array
using CUDA # CUDA must be loaded for cuFINUFFT to be activated

# Copy input data to GPU, "_d" suffix indiciates data on device (GPU)
x_d = CuArray(x);
c_d = CuArray(c);
# Allocate CUDA aray
out_d = CuArray{Complex{dtype}}(undef, ms);
# Note: identical interface as CPU, but with CUDA arrays on device

@btime nufft1d1!(x_d, c_d, 1, tol, out_d)       # 0.28 sec on A100 (3.6G NUpt/s)
# 20x the CPU speed, float32 or 64.

# Copy results back to host memory
gpu_results = Array(out_d);
magnitude = norm(fk, Inf)
@show norm(gpu_results-fk, Inf) / magnitude     # Should be < epsilon

