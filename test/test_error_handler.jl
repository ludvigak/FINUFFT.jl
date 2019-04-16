using FINUFFT
using Test

@testset "Error handling" begin

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
    opts.spread_kerevalmeth = 0
    opts.upsampfac = maxintfloat(typeof(opts.upsampfac))
    try 
        nufft1d1(xj, cj, iflag, tol, ms, opts)
    catch e
        @test e.errno == FINUFFT.ERR_MAXNALLOC
    end

    # Too small upsampfac
    opts = finufft_default_opts()
    opts.spread_kerevalmeth = 0
    opts.upsampfac = 0
    try 
        nufft1d1(xj, cj, iflag, tol, ms, opts)
    catch e
        @test e.errno == FINUFFT.ERR_UPSAMPFAC_TOO_SMALL
    end
    
    # Wrong beta
    opts = finufft_default_opts()
    opts.upsampfac = 0
    try 
        nufft1d1(xj, cj, iflag, tol, ms, opts)
    catch e
        @test e.errno == FINUFFT.HORNER_WRONG_BETA
    end
    
end
