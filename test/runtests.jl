using FINUFFT
using Base.Test

nj = 10
nk = 11
ms = 12
mt = 13
mu = 14

srand(1)

# nonuniform data
x = 3*pi*(1-2*rand(nj))
y = 3*pi*(1-2*rand(nj))
z = 3*pi*(1-2*rand(nj))
c = rand(nj) + 1im*rand(nj)
s = rand(nk)
t = rand(nk)
u = rand(nk)
f = rand(nk) + 1im*rand(nk)

# uniform data
F1D = rand(ms) + 1im*rand(ms)
F2D = rand(ms, mt) + 1im*rand(ms,mt)
F3D = rand(ms, mt, mu) + 1im*rand(ms,mt, mu)

modevec(m) = -floor(m/2):floor((m-1)/2+1)
k1 = modevec(ms)
k2 = modevec(mt)
k3 = modevec(mu)

@testset "NUFFT" begin
    ## 1D
    @testset "1D" begin
        # 1D1
        out = complex(zeros(ms))
        ref = complex(zeros(ms))
        for j=1:nj
            for ss=1:ms
                ref[ss] += c[j] * exp(1im*k1[ss]*x[j])
            end
        end
        nufft1d1!(x, c, 1, eps(), ms, out)
        relerr_1d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_1d1 < 1e-13
        
        # 1D2
        out = complex(zeros(nj))
        ref = complex(zeros(nj))
        for j=1:nj
            for ss=1:ms
                ref[j] += F1D[ss] * exp(1im*k1[ss]*x[j])
            end
        end
        nufft1d2!(x, out, 1, eps(), F1D)
        relerr_1d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_1d2 < 1e-13
        
        # 1D3
        out = complex(zeros(nk))
        ref = complex(zeros(nk))
        for k=1:nk
            for j=1:nj
                ref[k] += c[j] * exp(1im*s[k]*x[j])
            end
        end
        nufft1d3!(x,c,1,eps(),s,out)
        relerr_1d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_1d3 < 1e-13    
    end

    ## 2D
    @testset "2D" begin
        # 2D1
        out = complex(zeros(ms, mt))
        ref = complex(zeros(ms, mt))
        for j=1:nj
            for ss=1:ms
                for tt=1:mt
                    ref[ss,tt] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                end
            end
        end
        nufft2d1!(x, y, c, 1, eps(), ms, mt, out)
        relerr_2d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_2d1 < 1e-13
        
        # 2D2
        out = complex(zeros(nj))
        ref = complex(zeros(nj))
        for j=1:nj
            for ss=1:ms
                for tt=1:mt
                    ref[j] += F2D[ss, tt] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                end
            end
        end
        nufft2d2!(x, y, out, 1, eps(), F2D)
        relerr_2d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_2d2 < 1e-13

        # 2D3
        out = complex(zeros(nk))
        ref = complex(zeros(nk))
        for k=1:nk
            for j=1:nj
                ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]))
            end
        end
        nufft2d3!(x,y,c,1,eps(),s,t,out)
        relerr_2d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_2d3 < 1e-13
        
    end

    ## 3D
    @testset "3D" begin
        # 3D1
        out = complex(zeros(ms, mt, mu))
        ref = complex(zeros(ms, mt, mu))
        for j=1:nj
            for ss=1:ms
                for tt=1:mt
                    for uu=1:mu
                        ref[ss,tt,uu] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]+k3[uu]*z[j]))
                    end
                end
            end
        end
        nufft3d1!(x, y, z, c, 1, eps(), ms, mt, mu, out)
        relerr_3d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_3d1 < 1e-13

        # 3D2
        out = complex(zeros(nj))
        ref = complex(zeros(nj))
        for j=1:nj
            for ss=1:ms
                for tt=1:mt
                    for uu=1:mu
                        ref[j] += F3D[ss, tt, uu] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]+k3[uu]*z[j]))
                    end
                end
            end
        end
        nufft3d2!(x, y, z, out, 1, eps(), F3D)
        relerr_3d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_3d2 < 1e-13

        # 3D3
        out = complex(zeros(nk))
        ref = complex(zeros(nk))
        for k=1:nk
            for j=1:nj
                ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]+u[k]*z[j]))
            end
        end
        nufft3d3!(x,y,z,c,1,eps(),s,t,u,out)
        relerr_3d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
        @test relerr_3d3 < 1e-13    
    end
end
