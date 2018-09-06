__precompile__()

module FINUFFT
using PyCall

using Compat
using Compat.Libdl


### DIRECT INTERFACE

const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("FINUFFT is not properly installed. Please build it first.")
end



#const libfinufft = "/home/ludvig/local-workspace/finufft/lib/libfinufft.so"

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

## 1D

function finufft1d1_c(xj, cj, iflag, eps, fk,
                      opts=finufft_default_opts())
    nj = length(xj)
    ms = length(fk)    
    # Calling interface
    # int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts opts);
    ccall( (:finufft1d1_c, libfinufft),
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
end

function finufft1d2_c(xj, cj, iflag, eps, fk,
                      opts=finufft_default_opts())
    nj = length(xj)
    ms = length(fk)    
    # Calling interface
    # int finufft1d2_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts opts);
    ccall( (:finufft1d2_c, libfinufft),
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
end

function finufft1d3_c(xj, cj, iflag, eps, sk, fk,
                      opts=finufft_default_opts())
    nj = length(xj)
    nk = length(fk)    
    # Calling interface
    # int finufft1d3_c(int j,FLT* x,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT _Complex* f, nufft_c_opts opts);
    ccall( (:finufft1d3_c, libfinufft),
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
end


### PYTHON WRAPPER

nufft1d1! = PyNULL()
nufft1d2! = PyNULL()
nufft1d3! = PyNULL()
nufft2d1! = PyNULL()
nufft2d2! = PyNULL()
nufft2d3! = PyNULL()
nufft3d1! = PyNULL()
nufft3d2! = PyNULL()
nufft3d3! = PyNULL()

function __init__()
    finufftpy = pyimport("finufftpy")

    copy!(nufft1d1!, finufftpy[:nufft1d1])
    copy!(nufft1d2!, finufftpy[:nufft1d2])
    copy!(nufft1d3!, finufftpy[:nufft1d3])    
    
    copy!(nufft2d1!, finufftpy[:nufft2d1])
    copy!(nufft2d2!, finufftpy[:nufft2d2])
    copy!(nufft2d3!, finufftpy[:nufft2d3])

    copy!(nufft3d1!, finufftpy[:nufft3d1])
    copy!(nufft3d2!, finufftpy[:nufft3d2])
    copy!(nufft3d3!, finufftpy[:nufft3d3])

    #Libdl.dlopen("libgomp", Libdl.RTLD_GLOBAL)
    Libdl.dlopen("libfftw3_threads", Libdl.RTLD_GLOBAL)   
end

export nufft1d1!, nufft1d2!, nufft1d3!
export nufft2d1!, nufft2d2!, nufft2d3!
export nufft3d1!, nufft3d2!,  nufft3d3!

end # module
