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
export nufft_c_opts # backward-compability
export finufft_plan
export finufft_default_opts
export finufft_makeplan
export finufft_setpts
export finufft_exec
export finufft_destroy
export finufft_exec!
export BIGINT

## External dependencies
using finufft_jll

const libfinufft = finufft_jll.libfinufft

const BIGINT = Int64 # defined in include/dataTypes.h

const fftwReal = Union{Float64,Float32}

## FINUFFT opts struct from include/nufft_opts.h
"""
    mutable struct nufft_opts    
        debug              :: Cint
        spread_debug       :: Cint
        spread_sort        :: Cint
        spread_kerevalmeth :: Cint
        spread_kerpad      :: Cint
        chkbnds            :: Cint
        fftw               :: Cint
        modeord            :: Cint
        upsampfac          :: Cdouble
        spread_thread      :: Cint
        maxbatchsize       :: Cint
        nthreads           :: Cint
        showwarn           :: Cint
    end

Options struct passed to the FINUFFT library.

# Fields

    debug :: Cint
0: silent, 1: text basic timing output

    spread_debug :: Cint
passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)

    spread_sort :: Cint
passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)

    spread_kerevalmeth :: Cint
passed to spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)

    spread_kerpad :: Cint
passed to spread_opts, 0: don't pad to mult of 4, 1: do

    chkbnds :: Cint
0: don't check if input NU pts in [-3pi,3pi], 1: do

    fftw :: Cint
0:`FFTW_ESTIMATE`, or 1:`FFTW_MEASURE` (slow plan but faster)

    modeord :: Cint
0: CMCL-style increasing mode ordering (neg to pos), or\\
1: FFT-style mode ordering (affects type-1,2 only)

    upsampfac :: Cdouble
upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)

    spread_thread :: Cint
for ntrans>1 only.\\
0:auto,\\
1: sequential multithreaded,\\
2: parallel singlethreaded (Melody),\\
3: nested multithreaded (Andrea).

    maxbatchsize :: Cint
// for ntrans>1 only. max blocking size for vectorized, 0 for auto-set

    spread_nthr_atomic :: Cint
if >=0, threads above which spreader OMP critical goes atomic
    spread_max_sp_size :: Cint
if >0, overrides spreader (dir=1) max subproblem size
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
    upsampfac          :: Cdouble
    spread_thread      :: Cint
    maxbatchsize       :: Cint
    spread_nthr_atomic :: Cint
    spread_max_sp_size :: Cint
    nufft_opts{T}() where T <: fftwReal = new{T}()
end

const nufft_c_opts = nufft_opts # backward compability

"""
    finufft_default_opts()

Return a [`nufft_opts`](@ref) struct with the default FINUFFT settings. Set up the double precision variant by default.\\
See: <https://finufft.readthedocs.io/en/latest/usage.html#options>
"""
finufft_default_opts() = finufft_default_opts(nufft_opts{Float64}())

function finufft_default_opts(opts::nufft_opts{T}) where T <: Float64

    ccall( (:finufft_default_opts, libfinufft),
            Nothing,
            (Ref{nufft_opts},),
            opts
            )

    return opts
end

function finufft_default_opts(opts::nufft_opts{T}) where T <: Float32
    ccall( (:finufftf_default_opts, libfinufft),
            Nothing,
            (Ref{nufft_opts},),
            opts
            )

    return opts
end

### Error handling
### Error code should match Error code in https://github.com/flatironinstitute/finufft/blob/master/include/defs.h
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
    # Check return value and output error messages
    if ret==0
        return
    elseif ret==ERR_EPS_TOO_SMALL
        msg = "requested tolerance epsilon too small"
    elseif ret==ERR_MAXNALLOC
        msg = "attemped to allocate internal arrays larger than MAX_NF (defined in common.h)"
    elseif ret==ERR_SPREAD_BOX_SMALL
        msg = "spreader: fine grid too small"
    elseif ret==ERR_SPREAD_PTS_OUT_RANGE
        msg = "spreader: if chkbnds=1, a nonuniform point out of input range [-3pi,3pi]^d"
    elseif ret==ERR_SPREAD_ALLOC
        msg = "spreader: array allocation error"
    elseif ret==ERR_SPREAD_DIR
        msg = "spreader: illegal direction (should be 1 or 2)"
    elseif ret==ERR_UPSAMPFAC_TOO_SMALL
        msg = "upsampfac too small (should be >1)"
    elseif ret==HORNER_WRONG_BETA
        msg = "upsampfac not a value with known Horner eval: currently 2.0 or 1.25 only"
    elseif ret==ERR_NDATA_NOTVALID
        msg = "ndata not valid (should be >= 1)"
    elseif ret==ERR_TYPE_NOTVALID
        msg = "undefined type, type should be 1, 2, or 3"
    elseif ret==ERR_DIM_NOTVALID
        msg = "dimension should be 1, 2, or 3"
    elseif ret==ERR_ALLOC
        msg = "allocation error"
    elseif ret==ERR_SPREAD_THREAD_NOTVALID
        msg = "spread thread not valid"
    else
        msg = "unknown error"
    end
    throw(FINUFFTError(ret, msg))
end



### validate sizes of inputs for setpts
function valid_setpts(type::Integer,
                      dim::Integer,
                      x::Array{T},
                      y::Array{T}=T[],
                      z::Array{T}=T[],
                      s::Array{T}=T[],
                      t::Array{T}=T[],
                      u::Array{T}=T[]) where T <: fftwReal
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
                   c::Array{Complex{T}}) where T <: fftwReal
    ntrans = Cint(length(c) / length(x))
    @assert ntrans*length(x) == length(c)
    return ntrans
end

### infer number of modes from fk array
function get_nmodes_from_fk(dim::Integer,
                            fk::Array{Complex{T}}) where T <: fftwReal
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
            @assert value <: fftwReal
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

include("guru.jl")
include("simple.jl")

end # module
