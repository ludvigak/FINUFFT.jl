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

    T = dtype
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

    # copy data to GPU
    x_d = CuArray(x)
    y_d = CuArray(y)
    z_d = CuArray(z)
    c_d = CuArray(c)
    s_d = CuArray(s)
    t_d = CuArray(t)
    u_d = CuArray(u)
    F1D_d = CuArray(F1D)
    F2D_d = CuArray(F2D)
    F3D_d = CuArray(F3D)

    errfac = 100       # allowed multiple of tol for errors rel to direct calc

    @testset "cuFINUFFT ($T)" begin
        ## 1D
        @testset "1D" begin
            # 1D1 - here we include tests of opts-setting, vectorized, guru...
            @testset "1D1" begin
                out_d = CUDA.zeros(Complex{T},ms)
                ref = zeros(Complex{T},ms)     # direct calc...
                for j=1:nj
                    for ss=1:ms
                        ref[ss] += c[j] * exp(1im*k1[ss]*x[j])
                    end
                end

                # Simple device mem, no alloc
                nufft1d1!(x_d, c_d, 1, tol, out_d)
                relerr_1d1! = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d1! < errfac*tol

                # guru1d1 device mem
                plan = cufinufft_makeplan(1,[ms;],1,1,tol,dtype=T)
                cufinufft_setpts!(plan,x_d)
                out2_d = cufinufft_exec(plan,c_d)
                cufinufft_destroy!(plan)
                relerr_guru_d = norm(vec(Array(out2_d))-vec(ref), Inf) / norm(vec(ref), Inf)
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

                # simple vectorized ("many")
                outstack_d = CUDA.zeros(Complex{T}, ms, ntrans)
                nufft1d1!(x_d,CuArray(cstack),1,tol,outstack_d)
                relerr_many = norm(vec(Array(outstack_d))-vec(refstack), Inf) / norm(vec(refstack), Inf)
                @test relerr_many < errfac*tol
            end

            # 1D2
            @testset "1D2" begin
                out_d = CUDA.zeros(Complex{T},nj)
                ref = zeros(Complex{T},nj)
                for j=1:nj
                    for ss=1:ms
                        ref[j] += F1D[ss] * exp(1im*k1[ss]*x[j])
                    end
                end
                nufft1d2!(x_d, out_d, 1, tol, F1D_d)
                relerr_1d2! = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d2! < errfac*tol
                # guru1d2
                plan = cufinufft_makeplan(2,[ms;],1,1,tol,dtype=T)
                cufinufft_setpts!(plan,x)
                out2 = cufinufft_exec(plan,F1D)
                cufinufft_destroy!(plan)
                relerr_1d2_guru = norm(vec(out2)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d2_guru < errfac*tol
            end

            # 1D3
            @testset "1D3" begin
                out_d = CUDA.zeros(Complex{T},nk)
                ref = zeros(Complex{T},nk)
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*s[k]*x[j])
                    end
                end
                nufft1d3!(x_d,c_d,1,tol,s_d,out_d)
                relerr_1d3 = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d3 < errfac*tol
            end

        end

        ## 2D
        @testset "2D" begin
            @testset "2D1" begin
                # 2D1
                out_d = CUDA.zeros(Complex{T},ms,mt)
                ref = zeros(Complex{T},ms,mt)
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[ss,tt] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                        end
                    end
                end
                nufft2d1!(x_d, y_d, c_d, 1, tol, out_d)
                relerr_2d1! = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d1! < errfac*tol
                # guru2d1
                plan = cufinufft_makeplan(1,[ms; mt],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y)
                out2 = cufinufft_exec(plan,c)
                cufinufft_destroy!(plan)
                relerr_2d1_guru = norm(vec(out2)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d1_guru < errfac*tol
            end

            @testset "2D2" begin
                # 2D2
                out_d = CUDA.zeros(Complex{T},nj)
                ref = zeros(Complex{T},nj)
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[j] += F2D[ss, tt] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                        end
                    end
                end
                nufft2d2!(x_d, y_d, out_d, 1, tol, F2D_d)
                relerr_2d2! = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d2! < errfac*tol
                # guru2d2
                plan = cufinufft_makeplan(2,[ms; mt],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y)
                out2 = cufinufft_exec(plan,F2D)
                cufinufft_destroy!(plan)
                relerr_2d2_guru = norm(vec(out2)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d2_guru < errfac*tol
            end

            @testset "2D3" begin
                # 2D3
                out_d = CUDA.zeros(Complex{T},nk)
                ref = zeros(Complex{T},nk)
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]))
                    end
                end
                nufft2d3!(x_d,y_d,c_d,1,tol,s_d,t_d,out_d)
                relerr_2d3 = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d3 < errfac*tol
            end
        end

        ## 3D
        @testset "3D" begin
            @testset "3D1" begin
                # 3D1
                out_d = CUDA.zeros(Complex{T},ms,mt,mu)
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
                nufft3d1!(x_d, y_d, z_d, c_d, 1, tol, out_d)
                relerr_3d1! = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d1! < errfac*tol
                # guru3d1
                plan = cufinufft_makeplan(1,[ms; mt; mu],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y, z)
                out2 = cufinufft_exec(plan,c)
                cufinufft_destroy!(plan)
                relerr_3d1_guru = norm(vec(out2)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d1_guru < errfac*tol
            end

            @testset "3D2" begin
                # 3D2
                out_d = CUDA.zeros(Complex{T},nj)
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
                nufft3d2!(x_d, y_d, z_d, out_d, 1, tol, F3D_d)
                relerr_3d2! = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d2! < errfac*tol
                # guru 3d2
                plan = cufinufft_makeplan(2,[ms; mt; mu],1,1,tol,dtype=T)
                cufinufft_setpts!(plan, x, y, z)
                out2 = cufinufft_exec(plan,F3D)
                cufinufft_destroy!(plan)
                relerr_3d2_guru = norm(vec(out2)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d2_guru < errfac*tol
            end

            @testset "3D3" begin
                # 3D3
                out_d = CUDA.zeros(Complex{T},nk)
                ref = zeros(Complex{T},nk)
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]+u[k]*z[j]))
                    end
                end
                nufft3d3!(x_d,y_d,z_d,c_d,1,tol,s_d,t_d,u_d,out_d)
                relerr_3d3 = norm(vec(Array(out_d))-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d3 < errfac*tol
            end
        end
    end
end

# Main: do the tests if we can
if !CUDA.functional()
    @warn "CUDA not available, skipping cuFINUFFT tests"
elseif !FINUFFT.cufinufft_jll.is_available()
    @warn "cuFINUFFT not available, skipping cuFINUFFT tests"
else
    test_cuda(1e-14, Float64)
    test_cuda(1e-4, Float32)
end
;
