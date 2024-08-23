# threading tester for FINUFFT.jl

using FINUFFT

using Test
using LinearAlgebra
using Random

function test_nufft_in_threads(tol::Real, dtype::DataType)
    @assert dtype <: FINUFFT.finufftReal

    rng = MersenneTwister(1)

    T = dtype    # abbrev; we no longer infer dtype as type of tol
    # (this would be confusing since tol can be any type)
    nj = 10      # sizes for this small test: # NU pts
    nk = 11      # targ NU pts for t3
    ms = 12      # modes x
    mt = 13      # modes y
    mu = 14      # modes z

    # nonuniform data, using the full allowed input domain [-3pi,3pi)
    x = Array{T}(3 * pi * (2 * rand(rng, nj) .- 1.0))
    y = Array{T}(3 * pi * (2 * rand(rng, nj) .- 1.0))
    z = Array{T}(3 * pi * (2 * rand(rng, nj) .- 1.0))
    c = rand(rng, Complex{T}, nj)
    s = rand(rng, T, nk)
    t = rand(rng, T, nk)
    u = rand(rng, T, nk)
    f = rand(rng, Complex{T}, nk)

    # uniform data
    F1D = rand(rng, Complex{T}, ms)
    F2D = rand(rng, Complex{T}, ms, mt)
    F3D = rand(rng, Complex{T}, ms, mt, mu)

    modevec(m) = -floor(m / 2):floor((m - 1) / 2 + 1)
    k1 = modevec(ms)
    k2 = modevec(mt)
    k3 = modevec(mu)

    errfac = 100       # allowed multiple of tol for errors rel to direct calc
    errdifffac = 10    # allowed multiple of tol for errors rel to 2nd NUFFT

    @testset "Test FINUFF thread-safety in 1D ($T)" begin
        ## 1D
        @testset "1D Type 1" begin

            ref = zeros(Complex{T}, ms)     # direct calc...
            for j = 1:nj
                for ss = 1:ms
                    ref[ss] += c[j] * exp(1im * k1[ss] * x[j])
                end
            end

            iters = 100
            flag1 = falses(iters)
            flag2 = falses(iters)
            flag3 = falses(iters)


            Threads.@threads for i = 1:iters
                out = zeros(Complex{T}, ms)
                # Simple, writing into array, setting some non-default opts...
                nufft1d1!(x, c, 1, tol, out, debug=0, spread_sort=0, nthreads=1)
                relerr_1d1 = norm(vec(out) - vec(ref), Inf) / norm(vec(ref), Inf)
                flag1[i] = relerr_1d1 < errfac * tol

                # Different caller which returns array
                out2 = nufft1d1(x, c, 1, tol, ms, nthreads=1)
                reldiff = norm(vec(out) - vec(out2), Inf) / norm(vec(out), Inf)
                flag2[i] = reldiff < errdifffac * tol

                # Guru interface
                plan = finufft_makeplan(1, [ms;], 1, 1, tol, dtype=T, nthreads=1)
                finufft_setpts!(plan, x)
                out3 = finufft_exec(plan, c)
                finufft_destroy!(plan)
                relerr_guru = norm(vec(out3) - vec(ref), Inf) / norm(vec(ref), Inf)
                flag3[i] = relerr_guru < errfac * tol
            end

            @test all(flag1)
            @test all(flag2)
            @test all(flag3)
        end
    end
end

test_nufft_in_threads(1e-14, Float64)
test_nufft_in_threads(1e-4, Float32)