### Error handling

# Following should match error codes in https://github.com/flatironinstitute/finufft/blob/master/include/finufft_errors.h

const WARN_EPS_TOO_SMALL            = 1
const ERR_MAXNALLOC                 = 2
const ERR_SPREAD_BOX_SMALL          = 3
const ERR_SPREAD_PTS_OUT_RANGE      = 4 # DEPRECATED
const ERR_SPREAD_ALLOC              = 5
const ERR_SPREAD_DIR                = 6
const ERR_UPSAMPFAC_TOO_SMALL       = 7
const ERR_HORNER_WRONG_BETA         = 8
const ERR_NTRANS_NOTVALID           = 9
const ERR_TYPE_NOTVALID             = 10
const ERR_ALLOC                     = 11
const ERR_DIM_NOTVALID              = 12
const ERR_SPREAD_THREAD_NOTVALID    = 13
const ERR_NDATA_NOTVALID            = 14
const ERR_CUDA_FAILURE              = 15
const ERR_PLAN_NOTVALID             = 16
const ERR_METHOD_NOTVALID           = 17
const ERR_BINSIZE_NOTVALID          = 18
const ERR_INSUFFICIENT_SHMEM        = 19
const ERR_NUM_NU_PTS_INVALID        = 20
const ERR_LOCK_FUNS_INVALID         = 22
const ERR_SPREADONLY_UPSAMP_INVALID = 23

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
    elseif ret==WARN_EPS_TOO_SMALL
        msg = "requested tolerance epsilon too small to achieve (warning only)"
        @warn msg
        return
    elseif ret==ERR_MAXNALLOC
        msg = "stopped due to needing internal array size >MAX_NF (defined in defs.h)"
    elseif ret==ERR_SPREAD_BOX_SMALL
        msg = "spreader: fine grid too small compared to spread (kernel) width"
    elseif ret==ERR_SPREAD_PTS_OUT_RANGE
        msg = "spreader: [DEPRECATED]"
    elseif ret==ERR_SPREAD_ALLOC
        msg = "spreader: array allocation error"
    elseif ret==ERR_SPREAD_DIR
        msg = "spreader: illegal direction (should be 1 or 2)"
    elseif ret==ERR_UPSAMPFAC_TOO_SMALL
        msg = "upsampfac too small (should be >1.0)"
    elseif ret==ERR_HORNER_WRONG_BETA
        msg = "upsampfac not a value with known Horner poly eval rule (currently 2.0 or 1.25 only)"
    elseif ret==ERR_NTRANS_NOTVALID
        msg = "ntrans not valid in \"many\" (vectorized) or guru interface (should be >= 1)"
    elseif ret==ERR_TYPE_NOTVALID
        msg = "transform type invalid"
    elseif ret==ERR_ALLOC
        msg = "general internal allocation failure"
    elseif ret==ERR_DIM_NOTVALID
        msg = "dimension invalid"
    elseif ret==ERR_SPREAD_THREAD_NOTVALID
        msg = "spread_thread option invalid"
    elseif ret==ERR_NDATA_NOTVALID
        msg = "invalid mode array (more than ~2^31 modes, dimension with 0 modes, etc)"
    elseif ret==ERR_CUDA_FAILURE
        msg = "CUDA failure (failure to call any cuda function/kernel, malloc/memset, etc))"
    elseif ret==ERR_PLAN_NOTVALID
        msg = "attempt to destroy an uninitialized plan"
    elseif ret==ERR_METHOD_NOTVALID
        msg = "invalid spread/interp method for dim (attempt to blockgather in 1D, e.g.)"
    elseif ret==ERR_BINSIZE_NOTVALID
        msg = "size of bins for subprob/blockgather invalid"
    elseif ret==ERR_INSUFFICIENT_SHMEM
        msg = "GPU shmem too small for subprob/blockgather parameters"
    elseif ret==ERR_NUM_NU_PTS_INVALID
        msg = "invalid number of nonuniform points: nj or nk negative, or too big (see defs.h)"
    elseif ret==ERR_SPREADONLY_UPSAMP_INVALID
        msg = "invalid upsampfac set while using gpu_spreadinterponly mode"
    else
        msg = "error of type unknown to Julia interface! Check FINUFFT documentation"
    end
    throw(FINUFFTError(ret, msg))
end
