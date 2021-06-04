using FINUFFT

using Test
using LinearAlgebra
using Random


function test_nufft(tol::T) where T <: FINUFFT.fftwReal
    Random.seed!(1)

    nj = 10
    nk = 11
    ms = 12
    mt = 13
    mu = 14

    arrT = Array{T}

    # nonuniform data
    x = arrT(3*pi*(1.0 .- 2*rand(nj)))
    y = arrT(3*pi*(1.0 .- 2*rand(nj)))
    z = arrT(3*pi*(1.0 .- 2*rand(nj)))
    c = rand(Complex{T},nj)
    s = rand(T,nk)
    t = rand(T,nk)
    u = rand(T,nk)
    f = rand(Complex{T},nk)

    # uniform data
    F1D = rand(Complex{T}, ms)
    F2D = rand(Complex{T}, ms, mt)
    F3D = rand(Complex{T}, ms, mt, mu)

    modevec(m) = -floor(m/2):floor((m-1)/2+1)
    k1 = modevec(ms)
    k2 = modevec(mt)
    k3 = modevec(mu)


    @testset "NUFFT ($T)" begin
        ## 1D
        @testset "1D" begin
            # 1D1
            @testset "1D1" begin
                out = zeros(Complex{T},ms)
                ref = zeros(Complex{T},ms)
                for j=1:nj
                    for ss=1:ms
                        ref[ss] += c[j] * exp(1im*k1[ss]*x[j])
                    end
                end
                # Try this one with explicit opts struct
                nufft1d1!(x, c, 1, tol, out, debug=1, spread_kerpad=0)
                relerr_1d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d1 < 100tol
    #            # Different caller
                out2 = nufft1d1(x, c, 1, tol, ms, debug=1, spread_kerpad=0)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol

                #guru1d1
                plan = finufft_makeplan(1,[ms;],1,1,tol,spread_debug=1,debug=1)
                finufft_setpts(plan,x)
                out3 = finufft_exec(plan,c)
                finufft_destroy(plan)
                relerr_guru = norm(vec(out3)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_guru < 10tol
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
                nufft1d2!(x, out, 1, tol, F1D)
                relerr_1d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d2 < 100tol
                out2 = nufft1d2(x, 1, tol, F1D)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol
            end
            
            # 1D3
            @testset "1D3" begin
                out = zeros(Complex{T},nk)
                ref = zeros(Complex{T},nk)
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*s[k]*x[j])
                    end
                end
                nufft1d3!(x,c,1,tol,s,out)
                relerr_1d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d3 < 100tol
                out2 = nufft1d3(x,c,1,tol,s)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol
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
                nufft2d1!(x, y, c, 1, tol, out)
                relerr_2d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d1 < 100tol
                out2 = nufft2d1(x, y, c, 1, tol, ms, mt)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol        
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
                nufft2d2!(x, y, out, 1, tol, F2D)
                relerr_2d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d2 < 100tol
                out2 = nufft2d2(x, y, 1, tol, F2D)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol        
            end

            @testset "3D3" begin
                # 2D3
                out = zeros(Complex{T},nk)
                ref = zeros(Complex{T},nk)
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]))
                    end
                end
                nufft2d3!(x,y,c,1,tol,s,t,out)
                relerr_2d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d3 < 100tol
                out2 = nufft2d3(x,y,c,1,tol,s,t)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol
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
                nufft3d1!(x, y, z, c, 1, tol, out)
                relerr_3d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d1 < 100tol
                out2 = nufft3d1(x, y, z, c, 1, tol, ms, mt, mu)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol
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
                nufft3d2!(x, y, z, out, 1, tol, F3D)
                relerr_3d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d2 < 100tol
                out2 = nufft3d2(x, y, z, 1, tol, F3D)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol
            end

            @testset "3D3" begin
                # 3D3
                out = zeros(Complex{T},nk)
                ref = zeros(Complex{T},nk)
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]+u[k]*z[j]))
                    end
                end        
                nufft3d3!(x,y,z,c,1,tol,s,t,u,out)
                relerr_3d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d3 < 100tol
                out2 = nufft3d3(x,y,z,c,1,tol,s,t,u)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 10tol
            end        
        end
    end
end

test_nufft(1e-15)
test_nufft(1f-4)