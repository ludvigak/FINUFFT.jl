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
    ms = 12      # modes x

    # nonuniform data, using the full allowed input domain [-3pi,3pi)
    x = Array{T}(3 * pi * (2 * rand(rng, nj) .- 1.0))
    c = rand(rng, Complex{T}, nj)

    modevec(m) = -floor(m / 2):floor((m - 1) / 2 + 1)
    k1 = modevec(ms)

    errfac = 100       # allowed multiple of tol for errors rel to direct calc

    @testset "Test FINUFFT thread-safety ($T)" begin
        ## 1D
        @testset "For now only 1D Type 1" begin

            ref = zeros(Complex{T}, ms)     # direct calc...
            for j = 1:nj
                for ss = 1:ms
                    ref[ss] += c[j] * cis(k1[ss] * x[j])
                end
            end

            iters = 100
            flag = falses(iters)
            
            Threads.@threads for i = 1:iters
                out = nufft1d1(x, c, 1, tol, ms, nthreads=1)
                relerr_1d1 = norm(vec(out) .- vec(ref), Inf) / norm(vec(ref), Inf)
                flag[i] = relerr_1d1 < errfac * tol
            end

            @test all(flag)
        end
    end
end

test_nufft_in_threads(1e-14, Float64)
test_nufft_in_threads(1e-4, Float32)
