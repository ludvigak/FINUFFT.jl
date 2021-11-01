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
        @test e.errno == 1          # code should match FINUFFT.jl
    end

    # Allocate too much
    opts = finufft_default_opts()
    spread_kerevalmeth = 0
    upsampfac = maxintfloat(typeof(opts.upsampfac))   # hack to alloc a lot
    try 
        nufft1d1(xj, cj, iflag, tol, ms, spread_kerevalmeth=spread_kerevalmeth, upsampfac=upsampfac)
    catch e
        @test e.errno == 2
    end

    # Too small upsampfac
    spread_kerevalmeth = 0
    upsampfac = 0.9                     # note 0 is auto-choice
    try 
        nufft1d1(xj, cj, iflag, tol, ms, spread_kerevalmeth=spread_kerevalmeth, upsampfac=upsampfac)
    catch e
        @test e.errno == 7
    end
    
    # Wrong beta (ie upsampfac not known for Horner poly eval rule)
    upsampfac = 1.5
    try 
        nufft1d1(xj, cj, iflag, tol, ms, upsampfac=upsampfac)
    catch e
        @test e.errno == 8
    end
    
end
