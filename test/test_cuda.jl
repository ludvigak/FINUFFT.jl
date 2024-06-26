# test cuFINUFFT routines

using FINUFFT

using CUDA
using Test
using LinearAlgebra
using Random

# run a test at requested tolerance tol, floating-point precision dtype...
function test_cuda(tol::Real, dtype::DataType)
    @assert dtype <: FINUFFT.finufftReal

    # Warning test
    cuopt = FINUFFT.cufinufft_default_opts()
    @test_logs FINUFFT.setkwopts!(cuopt, modeord=1)
    @test_logs (:warn, "FINUFFT.cufinufft_opts does not have attribute foo") FINUFFT.setkwopts!(cuopt, foo=1)

    rng = MersenneTwister(1)

    T = dtype    # abbrev; we no longer infer dtype as type of tol
    # (this would be confusing since tol can be any type)
    nj = 10      # sizes for this small test: # NU pts
    nk = 11      # targ NU pts for t3
    ms = 12      # modes x
    mt = 13      # modes y
    mu = 14      # modes z

    # nonuniform data, using the full allowed input domain [-3pi,3pi)
    x = Array{T}(3*pi*(2*rand(rng, nj).-1.0))
    y = Array{T}(3*pi*(2*rand(rng, nj).-1.0))
    z = Array{T}(3*pi*(2*rand(rng, nj).-1.0))
    c = rand(rng, Complex{T},nj)
    s = rand(rng, T,nk)
    t = rand(rng, T,nk)
    u = rand(rng, T,nk)
    f = rand(rng, Complex{T},nk)

    # uniform data
    F1D = rand(rng, Complex{T}, ms)
    F2D = rand(rng, Complex{T}, ms, mt)
    F3D = rand(rng, Complex{T}, ms, mt, mu)

    modevec(m) = -floor(m/2):floor((m-1)/2+1)
    k1 = modevec(ms)
    k2 = modevec(mt)
    k3 = modevec(mu)

    errfac = 100       # allowed multiple of tol for errors rel to direct calc

    @testset "cuFINUFFT ($T)" begin
        ## 1D
        @testset "1D" begin
            # 1D1 - here we include tests of opts-setting, vectorized, guru...
            @testset "1D1" begin
                out = zeros(Complex{T},ms)
                ref = zeros(Complex{T},ms)     # direct calc...
                for j=1:nj
                    for ss=1:ms
                        ref[ss] += c[j] * exp(1im*k1[ss]*x[j])
                    end
                end

                # guru1d1 device mem
                plan = cufinufft_makeplan(1,[ms;],1,1,tol,dtype=T)
                x_d = CuArray(x)
                cufinufft_setpts!(plan,x_d)
                c_d = CuArray(c)
                out_d = cufinufft_exec(plan,c_d)
                cufinufft_destroy!(plan)
                relerr_guru_d = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_guru_d < errfac*tol

                # guru1d1
                plan = cufinufft_makeplan(1,[ms;],1,1,tol,dtype=T)
                cufinufft_setpts!(plan,x)
                out3 = cufinufft_exec(plan,c)
                cufinufft_destroy!(plan)
                relerr_guru = norm(vec(out3)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_guru < errfac*tol

                # guru1d1 vectorized ("many")
                ntrans = 3         # let's stack 3 transforms at once
                plan = cufinufft_makeplan(1,[ms;],1,ntrans,tol,dtype=T)
                cufinufft_setpts!(plan,x)
                cstack = hcat(c,2*c,3*c);           # change the coeff vectors
                out4 = cufinufft_exec(plan,cstack)
                cufinufft_destroy!(plan)
                refstack = hcat(ref,2*ref,3*ref);   # ditto
                relerr_guru_many = norm(vec(out4)-vec(refstack), Inf) / norm(vec(refstack), Inf)
                @test relerr_guru_many < errfac*tol
            end

            # 1D2
            @testset "1D2" begin
                out = zeros(Complex{T},nj)
                ref = zeros(Complex{T},nj)
                for j=1:nj
                    for ss=1:ms
                        ref[j] += F1D[ss] * exp(1im*k1[ss]*x[j])
                    end
                end

                # guru1d2
                plan = cufinufft_makeplan(2,[ms;],1,1,tol,dtype=T)
                cufinufft_setpts!(plan,x)
                out = cufinufft_exec(plan,F1D)
                cufinufft_destroy!(plan)

                relerr_1d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d2 < errfac*tol
            end
        end

        ## 2D
        @testset "2D" begin
            @testset "2D1" begin
                # 2D1
                out = zeros(Complex{T},ms,mt)
                ref = zeros(Complex{T},ms,mt)
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[ss,tt] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                        end
                    end
                end
                plan = cufinufft_makeplan(1,[ms; mt],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y)
                out = cufinufft_exec(plan,c)
                cufinufft_destroy!(plan)
                relerr_2d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d1 < errfac*tol
            end

            @testset "2D2" begin
                # 2D2
                out = zeros(Complex{T},nj)
                ref = zeros(Complex{T},nj)
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[j] += F2D[ss, tt] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                        end
                    end
                end
                plan = cufinufft_makeplan(2,[ms; mt],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y)
                out = cufinufft_exec(plan,F2D)
                cufinufft_destroy!(plan)
                relerr_2d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d2 < errfac*tol
            end
        end

        ## 3D
        @testset "3D" begin
            @testset "3D1" begin
                # 3D1
                out = zeros(Complex{T},ms,mt,mu)
                ref = zeros(Complex{T},ms,mt,mu)
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            for uu=1:mu
                                ref[ss,tt,uu] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]+k3[uu]*z[j]))
                            end
                        end
                    end
                end
                # guru3d1
                plan = cufinufft_makeplan(1,[ms; mt; mu],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y, z)
                out = cufinufft_exec(plan,c)
                cufinufft_destroy!(plan)
                relerr_guru = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_guru < errfac*tol
            end
            
            @testset "3D2" begin
                # 3D2
                out = zeros(Complex{T},nj)
                ref = zeros(Complex{T},nj)
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            for uu=1:mu
                                ref[j] += F3D[ss, tt, uu] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]+k3[uu]*z[j]))
                            end
                        end
                    end
                end
                plan = cufinufft_makeplan(2,[ms; mt; mu],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y, z)
                out = cufinufft_exec(plan,F3D)
                cufinufft_destroy!(plan)
                relerr_3d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d2 < errfac*tol
            end

        end
    end
end

# Main: do the tests
if !CUDA.functional()
    @warn "CUDA not available, skipping cuFINUFFT tests"
elseif !FINUFFT.cufinufft_jll.is_available()
    @warn "cuFINUFFT not available, skipping cuFINUFFT tests"
else
    test_cuda(1e-14, Float64)
    test_cuda(1e-4, Float32)
end
;
