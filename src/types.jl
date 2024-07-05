## Setup types
const BIGINT = Int64        # must match that in FINUFFT include/finufft/defs.h

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


# cufinufft_opts defined here to be available in helper defs
#
# This must match definition in
# finufft/include/cufinufft_opts.h
"""
    mutable struct cufinufft_opts
        upsampfac            :: Cdouble # upsampling ratio sigma, only 2.0 (standard) is implemented
        # following options are for gpu #
        gpu_method           :: Cint # 1: nonuniform-pts driven, 2: shared mem (SM)
        gpu_sort             :: Cint # when NU-pts driven: 0: no sort (GM), 1: sort (GM-sort)
        gpu_binsizex         :: Cint # used for 2D, 3D subproblem method
        gpu_binsizey         :: Cint
        gpu_binsizez         :: Cint
        gpu_obinsizex        :: Cint # used for 3D spread block gather method
        gpu_obinsizey        :: Cint
        gpu_obinsizez        :: Cint
        gpu_maxsubprobsize   :: Cint
        gpu_kerevalmeth      :: Cint # 0: direct exp(sqrt()), 1: Horner ppval
        gpu_spreadinterponly :: Cint # 0: NUFFT, 1: spread or interpolation only
        gpu_maxbatchsize     :: Cint
        # multi-gpu support #
        gpu_device_id        :: Cint
        gpu_stream           :: Ptr{Cvoid}
        modeord              :: Cint # (type 1,2 only): 0 CMCL-style increasing mode order
                                     #                  1 FFT-style mode order
    end

Options struct passed to cuFINUFFT, see C documentation.
"""
mutable struct cufinufft_opts
    upsampfac            :: Cdouble # upsampling ratio sigma, only 2.0 (standard) is implemented

    # following options are for gpu #
    gpu_method           :: Cint # 1: nonuniform-pts driven, 2: shared mem (SM)
    gpu_sort             :: Cint # when NU-pts driven: 0: no sort (GM), 1: sort (GM-sort)

    gpu_binsizex         :: Cint # used for 2D, 3D subproblem method
    gpu_binsizey         :: Cint
    gpu_binsizez         :: Cint

    gpu_obinsizex        :: Cint # used for 3D spread block gather method
    gpu_obinsizey        :: Cint
    gpu_obinsizez        :: Cint

    gpu_maxsubprobsize   :: Cint
    gpu_kerevalmeth      :: Cint # 0: direct exp(sqrt()), 1: Horner ppval

    gpu_spreadinterponly :: Cint # 0: NUFFT, 1: spread or interpolation only

    gpu_maxbatchsize     :: Cint

    # multi-gpu support #
    gpu_device_id        :: Cint

    gpu_stream           :: Ptr{Cvoid}

    modeord              :: Cint # (type 1,2 only): 0 CMCL-style increasing mode order
                                 #                  1 FFT-style mode order
    cufinufft_opts() = new()
end
