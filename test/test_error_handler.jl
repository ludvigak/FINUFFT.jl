using FINUFFT
using Test

@testset "Error handling and dumb inputs" begin

    xj = zeros(10)
    cj = complex(zeros(10))
    iflag = 1
    tol = 1e-15
    ms = 10

    @test_nowarn nufft1d1(xj, cj, iflag, tol, ms) # This should pass

    @info("Testing error handling")

    # Tolerance too small (should only warn)
    @test_warn "epsilon too small" nufft1d1(xj, cj, iflag, 1e-100, ms)

    # Allocate too much
    opts = finufft_default_opts()
    spread_kerevalmeth = 0
    upsampfac = maxintfloat(typeof(opts.upsampfac))   # hack to alloc a lot
    try
        nufft1d1(xj, cj, iflag, tol, ms, spread_kerevalmeth=spread_kerevalmeth, upsampfac=upsampfac)
    catch e
        @test e.errno == FINUFFT.ERR_MAXNALLOC
    end

    # Too small upsampfac
    spread_kerevalmeth = 0
    upsampfac = 0.9                     # note 0 is auto-choice
    try
        nufft1d1(xj, cj, iflag, tol, ms, spread_kerevalmeth=spread_kerevalmeth, upsampfac=upsampfac)
    catch e
        @test e.errno == FINUFFT.ERR_UPSAMPFAC_TOO_SMALL
    end

    # Wrong beta (ie upsampfac not known for Horner poly eval rule)
    upsampfac = 1.5
    try
        nufft1d1(xj, cj, iflag, tol, ms, upsampfac=upsampfac)
    catch e
        @test e.errno == FINUFFT.ERR_HORNER_WRONG_BETA
    end

    # Test immediate destroy and double-destroy, and their status codes...
    p = finufft_makeplan(2,10,+1,1,1e-6);
    @test finufft_destroy!(p)==0   # 0 signifies success.
    @test finufft_destroy!(p)==1   # 1 since already destroyed; watch for crash

    opt = finufft_default_opts(Float64)
    @test_nowarn FINUFFT.setkwopts!(opt, modeord=1)
    @test_warn "nufft_opts{Float64} does not have attribute foo" FINUFFT.setkwopts!(opt, foo=1)

    cuopt = FINUFFT.cufinufft_default_opts()
    @test_nowarn FINUFFT.setkwopts!(cuopt, modeord=1)
    @test_warn "FINUFFT.cufinufft_opts does not have attribute foo" FINUFFT.setkwopts!(cuopt, foo=1)

    @info("Error handling testing done")
end
