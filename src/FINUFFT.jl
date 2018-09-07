__precompile__()
module FINUFFT

## Export
export nufft1d1!, nufft1d2!, nufft1d3!
export nufft2d1!, nufft2d2!, nufft2d3!
export nufft3d1!, nufft3d2!,  nufft3d3!
export finufft_default_opts

## External dependencies
using Compat
using Compat.Libdl

const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("FINUFFT is not properly installed. Please build it first.")
end
function __init__()
    Libdl.dlopen("libfftw3_threads", Libdl.RTLD_GLOBAL)   
end

## FINUFFT opts struct from src/finufft_h.c
mutable struct nufft_c_opts  # see FINUFFT source common/finufft_default_opts() for defaults
    debug::Cint              # 0: silent, 1: text basic timing output
    spread_debug::Cint       # passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)
    spread_sort::Cint        # passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)
    spread_kerevalmeth::Cint # "     spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)
    spread_kerpad::Cint      # passed to spread_opts, 0: don't pad to mult of 4, 1: do
    chkbnds::Cint            # 0: don't check if input NU pts in [-3pi,3pi], 1: do
    fftw::Cint               # 0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)
    modeord::Cint            # 0: CMCL-style increasing mode ordering (neg to pos), or
                             # 1: FFT-style mode ordering (affects type-1,2 only)
    upsampfac::Cdouble       # upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
end

function finufft_default_opts()
    opts = nufft_c_opts(0,0,0,0,0,0,0,0,1.0)
    ccall( (:finufft_default_c_opts, libfinufft),
           Nothing,
           (Ref{nufft_c_opts},),
           opts
           )
    return opts
end

function check_ret(ret)
    # Check return value and output error messages
    if ret==0
        return
    elseif ret==1
        msg = "requested tolerance epsilon too small"
    elseif ret==2
        msg = "attemped to allocate internal arrays larger than MAX_NF (defined in common.h)"
    elseif ret==3
        msg = "spreader: fine grid too small"
    elseif ret==4
        msg = "spreader: if chkbnds=1, a nonuniform point out of input range [-3pi,3pi]^d"
    elseif ret==5
        msg = "spreader: array allocation error"
    elseif ret==6
        msg = "spreader: illegal direction (should be 1 or 2)"
    elseif ret==7
        msg = "upsampfac too small (should be >1)"
    elseif ret==8
        msg = "upsampfac not a value with known Horner eval: currently 2.0 or 1.25 only"
    elseif ret==9
        msg = "ndata not valid (should be >= 1)"
    else
        msg = "unknown error"
    end
    error("FINUFFT error: $msg")
end

## 1D

function nufft1d1!(xj, cj, iflag, eps, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    ms = length(fk)    
    # Calling interface
    # int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts opts);
    ret = ccall( (:finufft1d1_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, cj, iflag, eps, ms, fk, opts
                 )
    check_ret(ret)
end

function nufft1d2!(xj, cj, iflag, eps, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    ms = length(fk)    
    # Calling interface
    # int finufft1d2_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts opts);
    ret = ccall( (:finufft1d2_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, cj, iflag, eps, ms, fk, opts
                 )
    check_ret(ret)    
end

function nufft1d3!(xj, cj, iflag, eps, sk, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    @assert length(fk)==nk
    # Calling interface
    # int finufft1d3_c(int j,FLT* x,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT _Complex* f, nufft_c_opts opts);
    ret = ccall( (:finufft1d3_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, cj, iflag, eps, nk, sk, fk, opts
                 )
    check_ret(ret)
end

## 2D

function nufft2d1!(xj, yj, cj, iflag, eps, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    # Calling interface
    # int finufft2d1_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt,FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft2d1_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                 )
    check_ret(ret)
end

function nufft2d2!(xj, yj, cj, iflag, eps, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    # Calling interface
    # int finufft2d2_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft2d2_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                 )
    check_ret(ret)
end

function nufft2d3!(xj, yj, cj, iflag, eps, sk, tk, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(fk)==nk    
    # Calling interface
    # iint finufft2d3_c(int nj,FLT* x,FLT *y,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT _Complex* f, nufft_c_opts copts);
    ret = ccall( (:finufft2d3_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, cj, iflag, eps, nk, sk, tk, fk, opts
                 )
    check_ret(ret)
end

## 3D

function nufft3d1!(xj, yj, zj, cj, iflag, eps, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    # Calling interface
    # int finufft3d1_c(int nj,FLT* xj,FLT* yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu,FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft3d1_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},
                  Ref{Cdouble},                
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                 )
    check_ret(ret)
end

function nufft3d2!(xj, yj, zj, cj, iflag, eps, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    # Calling interface
    # int finufft3d2_c(int nj,FLT* xj,FLT *yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu, FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft3d2_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                 )
    check_ret(ret)
end

function nufft3d3!(xj, yj, zj, cj, iflag, eps, sk, tk, uk, fk,
                   opts=finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(uk)==nk    
    @assert length(fk)==nk    
    # Calling interface
    # int finufft3d3_c(int nj,FLT* x,FLT *y,FLT *z,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT *u,FLT _Complex* f, nufft_c_opts copts);
    ret = ccall( (:finufft3d3_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},
                  Ref{Cdouble},                  
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},
                  Ref{Cdouble},                        
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, zj, cj, iflag, eps, nk, sk, tk, uk, fk, opts
                 )
    check_ret(ret)
end

end # module
