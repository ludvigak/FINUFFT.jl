# medium-sized 2D type 1 transform, with timing test
using FINUFFT
using BenchmarkTools

nj = 10000000
x = pi*(1.0 .- 2.0*rand(nj));      # nonuniform points
y = pi*(1.0 .- 2.0*rand(nj));
c = rand(nj) + 1im*rand(nj);       # their strengths
n = 1000          # output size (number of Fourier modes per dim)
tol = 1e-9        # requested relative tolerance
out = Array{ComplexF64}(undef, n,n);     # prealloc
nufft2d1!(x, y, c, 1, tol, out);         # warm up / compile
@time nufft2d1!(x, y, c, 1, tol, out);   # crude time it
@btime nufft2d1!(x, y, c, 1, tol, out);  # better time it

# timing on 8-core AMD Ryzen 7 5700U laptop, tested Barnett 1/11/22...
#   precompiled binaries (finufft_jll.jl)  : 0.59 sec
#   locally compiled with GCC 10.3         : 0.35 sec         
