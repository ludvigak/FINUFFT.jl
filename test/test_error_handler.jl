using FINUFFT
using Test

@testset "Error handling and dumb inputs" begin

    xj = zeros(10)
    cj = complex(zeros(10))
    iflag = 1
    tol = 1e-15
    ms = 10
    
    nufft1d1(xj, cj, iflag, tol, ms) # This should pass

    @info("Testing error handling")

    # Tolerance too small
    try
        nufft1d1(xj, cj, iflag, 1e-100, ms)
    catch e
        @test e.errno == FINUFFT.ERR_EPS_TOO_SMALL
    end

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
        @test e.errno == FINUFFT.HORNER_WRONG_BETA
    end

    # Test immediate destroy and double-destroy, and their status codes...
    p = finufft_makeplan(2,10,+1,1,1e-6);
    @test finufft_destroy(p)==0   # 0 signifies success.
    @test finufft_destroy(p)==1   # 1 since already destroyed; watch for crash
end
