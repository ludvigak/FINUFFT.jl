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


### Guru Interfaces
finufft_plan_c = Ptr{Cvoid}

mutable struct finufft_plan{T}
    type       :: Cint
    ntrans     :: Cint
    dim        :: Cint
    ms         :: BIGINT
    mt         :: BIGINT
    mu         :: BIGINT
    nj         :: BIGINT
    nk         :: BIGINT
    plan_ptr   :: finufft_plan_c
end

### Double precision
function finufft_makeplan(type::Integer,
                          n_modes_or_dim::Union{Array{BIGINT},Integer},
                          iflag::Integer,
                          ntrans::Integer,
                          eps::T;
                          kwargs...) where T <: Float64
# see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
#   one can also use Array/Vector for cvoid pointer, Array and Ref both work
#   plan_p = Array{finufft_plan_c}(undef,1)
    plan_p = Ref{finufft_plan_c}()

    opts = finufft_default_opts(nufft_opts{T}())
    setkwopts!(opts;kwargs...)

    n_modes = ones(BIGINT,3)
    if type==3
        @assert ndims(n_modes_or_dim) == 0
        dim = n_modes_or_dim
    else
        @assert length(n_modes_or_dim)<=3 && length(n_modes_or_dim)>=1
        dim = length(n_modes_or_dim)
        n_modes[1:dim] .= n_modes_or_dim
    end

    ret = ccall( (:finufft_makeplan, libfinufft),
                    Cint,
                    (Cint,
                    Cint,
                    Ref{BIGINT},
                    Cint,
                    Cint,
                    Cdouble,
                    Ptr{finufft_plan_c},
                    Ref{nufft_opts}),
                    type,dim,n_modes,iflag,ntrans,eps,plan_p,opts
                    )
    
    check_ret(ret)

    ms = n_modes[1]
    mt = n_modes[2]
    mu = n_modes[3]
    plan = finufft_plan{T}(type,ntrans,dim,ms,mt,mu,0,0,plan_p[])
end

### Single precision
function finufft_makeplan(type::Integer,
                          n_modes_or_dim::Union{Array{BIGINT},Integer},
                          iflag::Integer,
                          ntrans::Integer,
                          eps::T;
                          kwargs...) where T <: Float32
# see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
#   one can also use Array/Vector for cvoid pointer, Array and Ref both work
#   plan_p = Array{finufft_plan_c}(undef,1)
    plan_p = Ref{finufft_plan_c}()

    opts = finufft_default_opts(nufft_opts{T}())
    setkwopts!(opts;kwargs...)

    n_modes = ones(BIGINT,3)
    if type==3
        @assert ndims(n_modes_or_dim) == 0
        dim = n_modes_or_dim
    else
        @assert length(n_modes_or_dim)<=3 && length(n_modes_or_dim)>=1
        dim = length(n_modes_or_dim)
        n_modes[1:dim] .= n_modes_or_dim
    end

    ret = ccall( (:finufftf_makeplan, libfinufft),
                      Cint,
                      (Cint,
                       Cint,
                       Ref{BIGINT},
                       Cint,
                       Cint,
                       Cfloat,
                       Ptr{finufft_plan_c},
                       Ref{nufft_opts}),
                      type,dim,n_modes,iflag,ntrans,eps,plan_p,opts
                      )

    check_ret(ret)

    ms = n_modes[1]
    mt = n_modes[2]
    mu = n_modes[3]
    plan = finufft_plan{T}(type,ntrans,dim,ms,mt,mu,0,0,plan_p[])
end

### Double precision
function finufft_setpts(plan::finufft_plan{T},
                        xj::Array{T},
                        yj::Array{T}=T[],
                        zj::Array{T}=T[],
                        s::Array{T}=T[],
                        t::Array{T}=T[],
                        u::Array{T}=T[]) where T <: Float64

    (M, N) = valid_setpts(plan.type, plan.dim, xj, yj, zj, s, t, u)

    plan.nj = M
    plan.nk = N

    ret = ccall( (:finufft_setpts, libfinufft),
                    Cint,
                    (finufft_plan_c,
                    BIGINT,
                    Ref{Cdouble},
                    Ref{Cdouble},
                    Ref{Cdouble},
                    BIGINT,
                    Ref{Cdouble},
                    Ref{Cdouble},
                    Ref{Cdouble}),
                    plan.plan_ptr,M,xj,yj,zj,N,s,t,u
                    )
    check_ret(ret)
    return ret
end

### Single precision
function finufft_setpts(plan::finufft_plan{T},
                        xj::Array{T},
                        yj::Array{T}=T[],
                        zj::Array{T}=T[],
                        s::Array{T}=T[],
                        t::Array{T}=T[],
                        u::Array{T}=T[]) where T <: Float32

    (M, N) = valid_setpts(plan.type, plan.dim, xj, yj, zj, s, t, u)

    plan.nj = M
    plan.nk = N

    ret = ccall( (:finufftf_setpts, libfinufft),
                    Cint,
                    (finufft_plan_c,
                    BIGINT,
                    Ref{Cfloat},
                    Ref{Cfloat},
                    Ref{Cfloat},
                    BIGINT,
                    Ref{Cfloat},
                    Ref{Cfloat},
                    Ref{Cfloat}),
                    plan.plan_ptr,M,xj,yj,zj,N,s,t,u
                    )
    check_ret(ret)
    return ret
end


function finufft_exec(plan::finufft_plan{T},
                      input::Array{Complex{T}}) where T <: fftwReal
    ret = 0
    type = plan.type
    ntrans = plan.ntrans
    dim = plan.dim
    n_modes = Array{BIGINT}(undef,3)
    n_modes[1] = plan.ms
    n_modes[2] = plan.mt
    n_modes[3] = plan.mu
    if type==1
        if dim==1
            output = Array{Complex{T}}(undef,n_modes[1],ntrans)
        elseif dim==2
            output = Array{Complex{T}}(undef,n_modes[1],n_modes[2],ntrans)
        elseif dim==3
            output = Array{Complex{T}}(undef,n_modes[1],n_modes[2],n_modes[3],ntrans)
        else
            ret = ERR_DIM_NOTVALID
        end
    elseif type==2
        nj = plan.nj
        output = Array{Complex{T}}(undef,nj,ntrans)
    elseif type==3
        nk = plan.nk
        output = Array{Complex{T}}(undef,nk,ntrans)
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
    finufft_exec!(plan,input,output)
    return output
end

function finufft_destroy(plan::finufft_plan{T}) where T <: Float64
    ret = ccall( (:finufft_destroy, libfinufft),
                 Cint,
                 (finufft_plan_c,),
                 plan.plan_ptr
                 )
    check_ret(ret)
    return ret
end

function finufft_destroy(plan::finufft_plan{T}) where T <: Float32
    ret = ccall( (:finufftf_destroy, libfinufft),
                    Cint,
                    (finufft_plan_c,),
                    plan.plan_ptr
                    )
    check_ret(ret)
    return ret
end

### Double precision
function finufft_exec!(plan::finufft_plan{T},
                      input::Array{Complex{T}},
                      output::Array{Complex{T}}) where T <: Float64
    type = plan.type
    ntrans = plan.ntrans
    dim = plan.dim
    n_modes = Array{BIGINT}(undef,3)
    n_modes[1] = plan.ms
    n_modes[2] = plan.mt
    n_modes[3] = plan.mu
    if type==1
        if dim==1
            if ntrans==1
                @assert size(output)==(n_modes[1],) || size(output)==(n_modes[1],ntrans)
            else
                @assert size(output)==(n_modes[1],ntrans)
            end
        elseif dim==2
            if ntrans==1
                @assert size(output)==(n_modes[1],n_modes[2]) || size(output)==(n_modes[1],n_modes[2],ntrans)
            else
                @assert size(output)==(n_modes[1],n_modes[2],ntrans)
            end
        elseif dim==3
            if ntrans==1
                @assert size(output)==(n_modes[1],n_modes[2],n_modes[3]) || size(output)==(n_modes[1],n_modes[2],n_modes[3],ntrans)
            else
                @assert size(output)==(n_modes[1],n_modes[2],n_modes[3],ntrans)
            end
        else
            ret = ERR_DIM_NOTVALID
            check_ret(ret)
        end
        ret = ccall( (:finufft_execute, libfinufft),
                        Cint,
                        (finufft_plan_c,
                        Ref{ComplexF64},
                        Ref{ComplexF64}),
                        plan.plan_ptr,input,output
                        )
    elseif type==2
        nj = plan.nj
        if ntrans==1
            @assert size(output)==(nj,ntrans) || size(output)==(nj,)
        else
            @assert size(output)==(nj,ntrans)
        end
        ret = ccall( (:finufft_execute, libfinufft),
                        Cint,
                        (finufft_plan_c,
                        Ref{ComplexF64},
                        Ref{ComplexF64}),
                        plan.plan_ptr,output,input
                        )
    elseif type==3
        nk = plan.nk
        if ntrans==1
            @assert size(output)==(nk,ntrans) || size(output)==(nk,)
        else
            @assert size(output)==(nk,ntrans)
        end
        ret = ccall( (:finufft_execute, libfinufft),
                        Cint,
                        (finufft_plan_c,
                        Ref{ComplexF64},
                        Ref{ComplexF64}),
                        plan.plan_ptr,input,output
                        )
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
end


### Single precision
function finufft_exec!(plan::finufft_plan{T},
                      input::Array{Complex{T}},
                      output::Array{Complex{T}}) where T <: Float32
    type = plan.type
    ntrans = plan.ntrans
    dim = plan.dim
    n_modes = Array{BIGINT}(undef,3)
    n_modes[1] = plan.ms
    n_modes[2] = plan.mt
    n_modes[3] = plan.mu
    if type==1
        if dim==1
            if ntrans==1
                @assert size(output)==(n_modes[1],) || size(output)==(n_modes[1],ntrans)
            else
                @assert size(output)==(n_modes[1],ntrans)
            end
        elseif dim==2
            if ntrans==1
                @assert size(output)==(n_modes[1],n_modes[2]) || size(output)==(n_modes[1],n_modes[2],ntrans)
            else
                @assert size(output)==(n_modes[1],n_modes[2],ntrans)
            end
        elseif dim==3
            if ntrans==1
                @assert size(output)==(n_modes[1],n_modes[2],n_modes[3]) || size(output)==(n_modes[1],n_modes[2],n_modes[3],ntrans)
            else
                @assert size(output)==(n_modes[1],n_modes[2],n_modes[3],ntrans)
            end
        else
            ret = ERR_DIM_NOTVALID
            check_ret(ret)
        end
        ret = ccall( (:finufftf_execute, libfinufft),
                        Cint,
                        (finufft_plan_c,
                        Ref{ComplexF32},
                        Ref{ComplexF32}),
                        plan.plan_ptr,input,output
                        )
    elseif type==2
        nj = plan.nj
        if ntrans==1
            @assert size(output)==(nj,ntrans) || size(output)==(nj,)
        else
            @assert size(output)==(nj,ntrans)
        end
        ret = ccall( (:finufftf_execute, libfinufft),
                        Cint,
                        (finufft_plan_c,
                        Ref{ComplexF32},
                        Ref{ComplexF32}),
                        plan.plan_ptr,output,input
                        )
    elseif type==3
        nk = plan.nk
        if ntrans==1
            @assert size(output)==(nk,ntrans) || size(output)==(nk,)
        else
            @assert size(output)==(nk,ntrans)
        end
        ret = ccall( (:finufftf_execute, libfinufft),
                        Cint,
                        (finufft_plan_c,
                        Ref{ComplexF32},
                        Ref{ComplexF32}),
                        plan.plan_ptr,input,output
                        )
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
end



### Simple Interfaces (allocate output)
### Double precision
## Type-1

"""
    nufft1d1(xj      :: Array{Float64} or Array{Float32}, 
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Float64 or Float32,
             ms      :: Integer;
             kwargs...
            ) -> Array{ComplexF64} or Array{ComplexF32}

Compute type-1 1D complex nonuniform FFT. 
"""
### should use type of xj to decide precision or not
function nufft1d1(xj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer,
                  eps::T,
                  ms::Integer;
                  kwargs...) where T <: fftwReal
    valid_setpts(1,1,xj)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, ms, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft1d1!(xj, cj, iflag, eps, fk; kwargs...)
    return fk
end

"""
    nufft2d1(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             ms      :: Integer,
             mt      :: Integer;
             kwargs...
            ) -> Array{ComplexF64}

Compute type-1 2D complex nonuniform FFT.
"""
function nufft2d1(xj      :: Array{T},
                  yj      :: Array{T},
                  cj      :: Array{Complex{T}},
                  iflag   :: Integer, 
                  eps     :: T,
                  ms      :: Integer,
                  mt      :: Integer;
                  kwargs...) where T <: fftwReal
    valid_setpts(1,2,xj,yj)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, ms, mt, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft2d1!(xj, yj, cj, iflag, eps, fk;kwargs...)
    return fk
end

"""
    nufft3d1(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             zj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             ms      :: Integer,
             mt      :: Integer,
             mu      :: Integer;
             kwargs...
            ) -> Array{ComplexF64}

Compute type-1 3D complex nonuniform FFT.
"""
function nufft3d1(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},
                  cj      :: Array{Complex{T}},
                  iflag   :: Integer, 
                  eps     :: T,
                  ms      :: Integer,
                  mt      :: Integer,
                  mu      :: Integer;
                  kwargs...)  where T <: fftwReal
    valid_setpts(1,3,xj,yj,zj)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, ms, mt, mu, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft3d1!(xj, yj, zj, cj, iflag, eps, fk;kwargs...)
    return fk
end


## Type-2

"""
    nufft1d2(xj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-2 1D complex nonuniform FFT. 
"""
function nufft1d2(xj      :: Array{T},
                  iflag   :: Integer,
                  eps     :: T,
                  fk      :: Array{Complex{T}};
                  kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(2,1,xj)
    (ms, ntrans) = get_nmodes_from_fk(1,fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft1d2!(xj, cj, iflag, eps, fk;kwargs...)
    return cj
end

"""
    nufft2d2(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-2 2D complex nonuniform FFT. 
"""
function nufft2d2(xj      :: Array{T},
                  yj      :: Array{T},
                  iflag   :: Integer,
                  eps     :: T,
                  fk      :: Array{Complex{T}};
                  kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(2,2,xj,yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2,fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft2d2!(xj, yj, cj, iflag, eps, fk;kwargs...)
    return cj
end

"""
    nufft3d2(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             zj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-2 3D complex nonuniform FFT. 
"""
function nufft3d2(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},
                  iflag   :: Integer, 
                  eps     :: T,
                  fk      :: Array{Complex{T}};
                  kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(2,3,xj,yj,zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3,fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft3d2!(xj, yj, zj, cj, iflag, eps, fk;kwargs...)
    return cj
end


## Type-3

"""
    nufft1d3(xj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             sk      :: Array{Float64};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-3 1D complex nonuniform FFT.
"""
function nufft1d3(xj      :: Array{T},
                  cj      :: Array{Complex{T}},
                  iflag   :: Integer, 
                  eps     :: T,
                  sk      :: Array{T};
                  kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(3,1,xj,T[],T[],sk)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft1d3!(xj, cj, iflag, eps, sk, fk;kwargs...)
    return fk
end

"""
    nufft2d3(xj      :: Array{Float64}, 
             yj      :: Array{Float64},
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             sk      :: Array{Float64},
             tk      :: Array{Float64};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-3 2D complex nonuniform FFT.
"""
function nufft2d3(xj      :: Array{T},
                  yj      :: Array{T}, 
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  sk      :: Array{T},
                  tk      :: Array{T};
                  kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(3,2,xj,yj,T[],sk,tk)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft2d3!(xj, yj, cj, iflag, eps, sk, tk, fk;kwargs...)
    return fk
end

"""
    nufft3d3(xj      :: Array{Float64}, 
             yj      :: Array{Float64},
             zj      :: Array{Float64},
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             sk      :: Array{Float64},
             tk      :: Array{Float64},
             uk      :: Array{Float64};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-3 3D complex nonuniform FFT.
"""
function nufft3d3(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},                   
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  sk      :: Array{T},
                  tk      :: Array{T},
                  uk      :: Array{T};
                  kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(3,3,xj,yj,zj,sk,tk,uk)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    # checkkwdtype(T; kwargs...)
    nufft3d3!(xj, yj, zj, cj, iflag, eps, sk, tk, uk, fk;kwargs...)
    return fk
end


### Direct interfaces (No allocation)
### Double precision
## 1D

"""
    nufft1d1!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64};
              kwargs...
            )

Compute type-1 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d1!(xj      :: Array{T},
                   cj      :: Array{Complex{T}},
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    valid_setpts(1,1,xj)
    ntrans = valid_ntr(xj,cj)
    (ms, ntrans_fk) = get_nmodes_from_fk(1,fk)

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(1,[ms;],iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
end


"""
    nufft1d2!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64};
              kwargs...
            )

Compute type-2 1D complex nonuniform FFT. Output stored in cj.
"""
function nufft1d2!(xj      :: Array{T},
                   cj      :: Array{Complex{T}},
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(2,1,xj)
    (ms, ntrans) = get_nmodes_from_fk(1,fk)

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(2,[ms;],iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj)
    finufft_exec!(plan,fk,cj)
    ret = finufft_destroy(plan)
    check_ret(ret)    
end


"""
    nufft1d3!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              fk      :: Array{ComplexF64};
              kwargs...
             )

Compute type-3 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d3!(xj      :: Array{T},
                   cj      :: Array{Complex{T}},
                   iflag   :: Integer, 
                   eps     :: T,
                   sk      :: Array{T},
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(3,1,xj,T[],T[],sk)
    ntrans = valid_ntr(xj,cj)

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(3,1,iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,T[],T[],sk)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
end


## 2D

"""
    nufft2d1!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64};
              kwargs...
            )

Compute type-1 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d1!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    valid_setpts(1,2,xj,yj)
    ntrans = valid_ntr(xj,cj)
    (ms, mt, ntrans_fk) = get_nmodes_from_fk(2,fk)
    @assert ntrans==ntrans_fk

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(1,[ms;mt],iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,yj)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
end


"""
    nufft2d2!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64};
              kwargs...
            )

Compute type-2 2D complex nonuniform FFT. Output stored in cj.
"""
function nufft2d2!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(1,2,xj,yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2,fk)

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(2,[ms;mt],iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,yj)
    finufft_exec!(plan,fk,cj)
    ret = finufft_destroy(plan)
    check_ret(ret)
end

"""
    nufft2d3!(xj      :: Array{Float64}, 
              yj      :: Array{Float64},
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              tk      :: Array{Float64},
              fk      :: Array{ComplexF64};
              kwargs...
             )

Compute type-3 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d3!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   sk      :: Array{T},
                   tk      :: Array{T},
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(3,2,xj,yj,T[],sk,tk)
    ntrans = valid_ntr(xj,cj)

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(3,2,iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,yj,T[],sk,tk)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
end

## 3D

"""
    nufft3d1!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              zj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64};
              kwargs...
            )

Compute type-1 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d1!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   zj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    valid_setpts(1,3,xj,yj,zj)
    ntrans = valid_ntr(xj,cj)
    (ms, mt, mu, ntrans_fk) = get_nmodes_from_fk(3,fk)
    @assert ntrans == ntrans_fk

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(1,[ms;mt;mu],iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,yj,zj)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
end

"""
    nufft3d2!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              zj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64};
              kwargs...
            )

Compute type-2 3D complex nonuniform FFT. Output stored in cj.
"""
function nufft3d2!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   zj      :: Array{T},                    
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(2,3,xj,yj,zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3,fk)

    # checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(2,[ms;mt;mu],iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,yj,zj)
    finufft_exec!(plan,fk,cj)
    ret = finufft_destroy(plan)
    check_ret(ret)
end

"""
    nufft3d3!(xj      :: Array{Float64}, 
              yj      :: Array{Float64},
              zj      :: Array{Float64},
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              tk      :: Array{Float64},
              uk      :: Array{Float64},
              fk      :: Array{ComplexF64};
              kwargs...
             )

Compute type-3 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d3!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   zj      :: Array{T},                   
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   sk      :: Array{T},
                   tk      :: Array{T},
                   uk      :: Array{T},
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: fftwReal
    (nj, nk) = valid_setpts(3,3,xj,yj,zj,sk,tk,uk)
    ntrans = valid_ntr(xj,cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(3,3,iflag,ntrans,eps;kwargs...)
    finufft_setpts(plan,xj,yj,zj,sk,tk,uk)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
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

end # module
