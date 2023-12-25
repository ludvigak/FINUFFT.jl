using FINUFFT
using Test
using JET: JET, @test_opt

@testset "Type stability and allocations" begin
    xj = zeros(10)
    cj = complex(zeros(10))
    iflag = 1
    tol = 1e-15
    ms = 10

    # Ignore JET errors due to CoreLogging (e.g. @warn)
    ignore = (JET.AnyFrameModule(Base.CoreLogging),)

    @test_opt ignored_modules=ignore nufft1d1(xj, cj, iflag, tol, ms)

    type = 1
    ntrans = 1
    out = similar(cj, ms)

    plan = finufft_makeplan(type, ms, iflag, ntrans, tol)
    finufft_setpts!(plan, xj)
    finufft_exec!(plan, cj, out)

    @test 0 == @allocated finufft_setpts!(plan, xj)
    @test 0 == @allocated finufft_exec!(plan, cj, out)
end
