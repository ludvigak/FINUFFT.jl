using FINUFFT

# Here we demo a Float64 1D type 1 transform
nj = 1000000
x = pi*(1.0 .- 2.0*rand(nj))      # nonuniform points
c = rand(nj) + 1im*rand(nj)       # their strengths

ms = 2000000      # output size (number of Fourier modes)
tol = 1e-9        # requested relative tolerance

# Output as return value (1e6 pts to 2e6 modes takes about 0.1 sec)...
fk = nufft1d1(x, c, 1, tol, ms)

# Or, output into preallocated array, whose size determines ms...
out = Array{ComplexF64}(undef, ms)
nufft1d1!(x, c, 1, tol, out)

# Demo using keyword args to change options from defaults...
fk_fftord = nufft1d1(x, c, 1, tol, ms, debug=1, modeord=1, nthreads=4)
