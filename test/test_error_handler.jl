using Logging
using FINUFFT
using Test

@testset "Error handling and dumb inputs" begin

    xj = zeros(10)
    cj = complex(zeros(10))
    iflag = 1
    tol = 1e-14
    ms = 10

    @test_logs nufft1d1(xj, cj, iflag, tol, ms) # Should not warn

    @info("Testing error handling")

    # Tolerance too small
    err =
        try
            nufft1d1(xj, cj, iflag, 1e-100, ms)
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno==FINUFFT.ERR_EPS_TOO_SMALL
    
    # Allocate too much
    opts = finufft_default_opts()
    upsampfac = maxintfloat(typeof(opts.upsampfac))   # hack to alloc a lot
    err = 
        try
            nufft1d1(xj, cj, iflag, tol, ms, upsampfac=upsampfac)
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno == FINUFFT.ERR_MAXNALLOC

    # Too small upsampfac
    upsampfac = 0.9                     # note 0 is auto-choice
    err = 
        try
            nufft1d1(xj, cj, iflag, tol, ms, upsampfac=upsampfac)
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno == FINUFFT.ERR_UPSAMPFAC_TOO_SMALL

    # Bad spread kernel formula
    err =
        try
            nufft1d1(xj, cj, iflag, tol, ms, spread_kerformula=1234)
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno==FINUFFT.ERR_KERFORMULA_NOTVALID

    # Invalid transform type
    err =
        try
            finufft_makeplan(4, ms, iflag, 1, tol)
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno == FINUFFT.ERR_TYPE_NOTVALID

    # Invalid number of transforms
    err =
        try
            finufft_makeplan(1, ms, iflag, 0, tol)
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno == FINUFFT.ERR_NTRANS_NOTVALID

    # Bad lock fun
    err =
        try
            nufft1d1(xj, cj, iflag, tol, ms, fftw_lock_fun=Ptr{Nothing}(0))
        catch e; e; end
    @test err isa FINUFFT.FINUFFTError
    @test err.errno==FINUFFT.ERR_LOCK_FUNS_INVALID

    # Test immediate destroy and double-destroy, and their status codes...
    p = finufft_makeplan(2,10,+1,1,1e-6);
    @test finufft_destroy!(p)==0   # 0 signifies success.
    @test finufft_destroy!(p)==1   # 1 since already destroyed; watch for crash

    opt = finufft_default_opts(Float64)
    @test_logs FINUFFT.setkwopts!(opt, modeord=1)
    @test_logs (:warn, "nufft_opts{Float64} does not have attribute foo") FINUFFT.setkwopts!(opt, foo=1)

    @info("Error handling testing done")
end
