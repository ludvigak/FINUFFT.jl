### Here we demo CUDA routines using the 1D type 1 transform

using FINUFFT
using CUDA # CUDA must be loaded for cuFINUFFT to be activated
using LinearAlgebra

dtype = Float64 # Datatype for computations
tol   = 1e-12   # requested relative tolerance

# Setup problem
nj = 10000000
x = pi*(1 .- 2*rand(dtype, nj)) # nonuniform points
c = rand(Complex{dtype}, nj)    # their strengths
ms = 20000                      # output size (number of Fourier modes)

# CPU computation
fk = nufft1d1(x, c, 1, tol, ms)

# CPU computation with preallocated array
out = Array{Complex{dtype}}(undef, ms)
nufft1d1!(x, c, 1, tol, out)

@show norm(out-fk, Inf) # Should be identical

##############################################
## Simple GPU interface for preallocated array

# Copy input data to GPU, "_d" suffix indiciates data on device (GPU)
x_d = CuArray(x)
c_d = CuArray(c)
# Allocate CUDA aray
out_d = CuArray{Complex{dtype}}(undef, ms)
# Note: identical interface as CPU, but with CUDA arrays on device
nufft1d1!(x_d, c_d, 1, tol, out_d)
# Copy results back to host memory
gpu_results = Array(out_d)

magnitude = norm(fk, Inf)
@show norm(gpu_results-fk, Inf) / magnitude # Should be < epsilon

##############################################
## GPU "guru" interface.
# This is what is implemented (and documented) in the base cuFINUFFT library

# Create plan
plan = cufinufft_makeplan(1,[ms;],1,1,tol, dtype=dtype)

# Set the nonuniform points to be used in the plan
cufinufft_setpts!(plan, x_d)
# this routine also accepts host data, in which case it will be automatically copied to device:
# cufinufft_setpts!(plan,x)

# Execute plan using allocated input and output arrays on device (recommended use)
cufinufft_exec!(plan, c_d, out_d)
# We can also call it without preallocated output, in which case ouput gets allocated for us (on device)
out2_d = cufinufft_exec(plan, c_d)
gpu_results2 = Array(out2_d)
# and if input data is on host, then output data is returned on host (after allcations and 2-way copying under the hood, which is costly)
gpu_results3 = cufinufft_exec(plan, c)

# These will vary to epsilon
@show norm(gpu_results - gpu_results2) / magnitude
@show norm(gpu_results - gpu_results3) / magnitude

# Finally we destroy the plan to have all internally allocated device memory freed. (This does not include returned data.)
cufinufft_destroy!(plan)
