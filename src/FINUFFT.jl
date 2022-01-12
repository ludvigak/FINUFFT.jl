__precompile__()
module FINUFFT

## Export
export nufft1d1, nufft1d2, nufft1d3
export nufft2d1, nufft2d2, nufft2d3
export nufft3d1, nufft3d2, nufft3d3

export nufft1d1!, nufft1d2!, nufft1d3!
export nufft2d1!, nufft2d2!, nufft2d3!
export nufft3d1!, nufft3d2!, nufft3d3!

export nufft_opts
export nufft_c_opts        # backward-compatibility - remove?
export finufft_plan
export finufft_default_opts
export finufft_makeplan
export finufft_setpts!
export finufft_exec
export finufft_destroy!
export finufft_exec!
export BIGINT
export finufftReal


# By default we depend on our precompiled generic binary package...
using finufft_jll
const libfinufft = finufft_jll.libfinufft
#
# If instead you want to use your locally-compiled FINUFFT library for more
# performance, comment out the above two code lines, uncomment the upcoming
# one, and edit it for the location of your local FINUFFT installation. You
# then need to use this FINUFFT.jl pkg in dev mode and restart (see README.md):
#const libfinufft = "/PATH/TO/YOUR/finufft/lib/libfinufft.so"


const BIGINT = Int64        # must match that in FINUFFT include/dataTypes.h

# our allowed real array types...
const finufftReal = Union{Float64,Float32}

## FINUFFT opts struct, must bytewise match that in include/nufft_opts.h
"""
    mutable struct nufft_opts    
        modeord            :: Cint
        chkbnds            :: Cint
        debug              :: Cint
        spread_debug       :: Cint
        showwarn           :: Cint
        nthreads           :: Cint
        fftw               :: Cint
        spread_sort        :: Cint
        spread_kerevalmeth :: Cint
        spread_kerpad      :: Cint
        upsampfac          :: Cdouble
        spread_thread      :: Cint
        maxbatchsize       :: Cint
        spread_nthr_atomic :: Cint
        spread_max_sp_size :: Cint
    end

Options struct passed to the FINUFFT library.

# Fields

This is a summary only; see FINUFFT documentation for full descriptions.

    modeord :: Cint
0: CMCL-style increasing mode ordering (neg to pos), or\\
1: FFT-style mode ordering (affects type-1,2 only)

    chkbnds :: Cint
0: don't check if input NU pts in [-3pi,3pi], 1: do

    debug :: Cint
0: silent, 1: text basic timing output

    spread_debug :: Cint
passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)

    showwarn :: Cint
Whether to print warnings to stderr. 0: silent, 1: print warnings
    
    nthreads :: Cint
How many threads FINUFFT should use, or 0 (use max available in OMP)

    fftw :: Cint
0:`FFTW_ESTIMATE`, or 1:`FFTW_MEASURE` (slow plan but faster FFTs)

    spread_sort :: Cint
passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)

    spread_kerevalmeth :: Cint
passed to spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)

    spread_kerpad :: Cint
passed to spread_opts, 0: don't pad to mult of 4, 1: do

    upsampfac :: Cdouble
upsampling ratio sigma: 2.0 (standard), or 1.25 (small FFT), or\\
0.0 (auto).

    spread_thread :: Cint
(for ntrans>1 only)\\
0: auto choice,\\
1: sequential multithreaded,\\
2: parallel singlethreaded spread.

    maxbatchsize :: Cint
(for ntrans>1 only). max blocking size for vectorized, 0 for auto-set

    spread_nthr_atomic :: Cint
if >=0, threads above which spreader OMP critical goes atomic

    spread_max_sp_size :: Cint
if >0, overrides spreader (dir=1 only) max subproblem size
"""
mutable struct nufft_opts{T}    
    modeord            :: Cint
    chkbnds            :: Cint
    debug              :: Cint
    spread_debug       :: Cint
    showwarn           :: Cint
    nthreads           :: Cint
    fftw               :: Cint
    spread_sort        :: Cint
    spread_kerevalmeth :: Cint
    spread_kerpad      :: Cint
    upsampfac          :: Cdouble        # regardless of T
    spread_thread      :: Cint
    maxbatchsize       :: Cint
    spread_nthr_atomic :: Cint
    spread_max_sp_size :: Cint
    nufft_opts{T}() where T <: finufftReal = new{T}()
end
# The above must match include/nufft_opts.h in FINUFFT.
# Note that nufft_opts happens to be the same for Float32 vs Float64 types.
# This is as in FINUFFT. It may not always hold in the future.

const nufft_c_opts = nufft_opts        # for backward compatibility - remove?

"""
    p = finufft_default_opts()
    p = finufft_default_opts(dtype=Float32)

Return a [`nufft_opts`](@ref) struct with the default FINUFFT settings. Set up the double precision variant by default.\\
See: <https://finufft.readthedocs.io/en/latest/usage.html#options>
"""
function finufft_default_opts(dtype::DataType=Float64)
    opts = nufft_opts{dtype}()

    if dtype==Float64
        ccall( (:finufft_default_opts, libfinufft),
               Nothing,
               (Ref{nufft_opts},),
               opts
               )

    else
        ccall( (:finufftf_default_opts, libfinufft),
               Nothing,
               (Ref{nufft_opts},),
               opts
               )
    end

    return opts
end

### Error handling

# Following should match error codes in https://github.com/flatironinstitute/finufft/blob/master/include/defs.h
const ERR_EPS_TOO_SMALL        = 1
const ERR_MAXNALLOC            = 2
const ERR_SPREAD_BOX_SMALL     = 3
const ERR_SPREAD_PTS_OUT_RANGE = 4
const ERR_SPREAD_ALLOC         = 5
const ERR_SPREAD_DIR           = 6
const ERR_UPSAMPFAC_TOO_SMALL  = 7
const HORNER_WRONG_BETA        = 8
const ERR_NDATA_NOTVALID       = 9
const ERR_TYPE_NOTVALID        = 10
const ERR_ALLOC                = 11
const ERR_DIM_NOTVALID         = 12
const ERR_SPREAD_THREAD_NOTVALID = 13

struct FINUFFTError <: Exception
    errno::Cint
    msg::String
end
Base.showerror(io::IO, e::FINUFFTError) = print(io, "FINUFFT Error ($(e.errno)): ", e.msg)

function check_ret(ret)
    # Check return value and output corresponding error message to Julia side.
    # This should be kepts up to date with error code interpretation in
    # https://finufft.readthedocs.io/en/latest/error.html
    # which is generated from
    # https://github.com/flatironinstitute/finufft/blob/master/docs/error.rst
    if ret==0     # no error or warning
        return
    elseif ret==ERR_EPS_TOO_SMALL
        msg = "requested tolerance epsilon too small to achieve (warning only)"
    elseif ret==ERR_MAXNALLOC
        msg = "attemped to allocate internal array larger than MAX_NF (defined in defs.h)"
    elseif ret==ERR_SPREAD_BOX_SMALL
        msg = "spreader: fine grid too small compared to spread (kernel) width"
    elseif ret==ERR_SPREAD_PTS_OUT_RANGE
        msg = "spreader: if chkbnds=1, a nonuniform point is out of input range [-3pi,3pi]^d"
    elseif ret==ERR_SPREAD_ALLOC
        msg = "spreader: array allocation error"
    elseif ret==ERR_SPREAD_DIR
        msg = "spreader: illegal direction (should be 1 or 2)"
    elseif ret==ERR_UPSAMPFAC_TOO_SMALL
        msg = "upsampfac too small (should be >1.0)"
    elseif ret==HORNER_WRONG_BETA
        msg = "upsampfac not a value with known Horner eval: currently 2.0 or 1.25 only"
    elseif ret==ERR_NDATA_NOTVALID
        msg = "ntrans not valid in vectorized interface (should be >= 1)"
    elseif ret==ERR_TYPE_NOTVALID
        msg = "invalid transform type, type should be 1, 2, or 3"
    elseif ret==ERR_ALLOC
        msg = "general allocation failure"
    elseif ret==ERR_DIM_NOTVALID
        msg = "invalid dimension, should be 1, 2, or 3"
    elseif ret==ERR_SPREAD_THREAD_NOTVALID
        msg = "spread_thread option not valid"
    else
        msg = "error of type unknown to Julia interface! Check FINUFFT documentation"
    end
    throw(FINUFFTError(ret, msg))
end


# HELPER routines for guru interface...

### validate sizes of inputs for setpts
function valid_setpts(type::Integer,
                      dim::Integer,
                      x::Array{T},
                      y::Array{T}=T[],
                      z::Array{T}=T[],
                      s::Array{T}=T[],
                      t::Array{T}=T[],
                      u::Array{T}=T[]) where T <: finufftReal
    nj = length(x)
    if type==3
        nk = length(s)
    else
        nk = 0
    end

    if dim>1
        @assert nj == length(y)
        if type==3
            @assert nk == length(t)
        end
    end

    if dim>2
        @assert nj == length(z)
        if type==3
            @assert nk == length(u)
        end
    end

    return (nj, nk)
end

### validate number of transforms
function valid_ntr(x::Array{T},
                   c::Array{Complex{T}}) where T <: finufftReal
    ntrans = Cint(length(c) / length(x))
    @assert ntrans*length(x) == length(c)
    return ntrans
end

### infer number of modes from fk array
function get_nmodes_from_fk(dim::Integer,
                            fk::Array{Complex{T}}) where T <: finufftReal
    ndim = ndims(fk)
    @assert dim==1 || dim==2 || dim==3
    @assert ndim==dim || ndim==dim+1
    if ndim==dim
        ntrans = 1
        return (size(fk)...,ntrans)
    else
        return size(fk)
    end
end

### kwargs opt set
function setkwopts!(opts::nufft_opts; kwargs...)
    dtype = Float64
    for (key, value) in kwargs
        if hasproperty(opts, key::Symbol)
            setproperty!(opts, key, value)
        elseif String(key)=="dtype"
            @assert value <: finufftReal
            dtype = value
        else
            @warn "nufft_opts does not have attribute " * String(key)
        end
    end
    return dtype
end

### check kwargs with dtype
function checkkwdtype(dtype::DataType; kwargs...)
    for (key, value) in kwargs
        if String(key)=="dtype"
            @assert  value == dtype
        end
    end
end


# Finally, bring in the main user-facing interfaces...
        
include("guru.jl")
include("simple.jl")

end # module
