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

"""
Creates a finufft_makeplan struct in the guru interface to FINUFFT, of
 type 1, 2 or 3, and with given numbers of Fourier modes (unless type 3).

Inputs:
    type            transform type: 1, 2, or 3
    n_modes_or_dim  if type is 1 or 2, the number of Fourier modes in each
                    dimension: [ms] in 1D, [ms mt] in 2D, or [ms mt mu] in 3D.
                    Its length sets the dimension, which must be 1, 2 or 3.
                    If type is 3, in contrast, its *value* fixes the dimension
    iflag   if >=0, uses + sign in exponential, otherwise - sign.
    eps     relative precision requested (generally between 1e-15 and 1e-1),
            real, need not match type of dtype
    ntrans          number of transforms to compute simultaneously
    dtype           Float32 or Float64, plan for single precision or double precision
    kwargs  (optional), for more options, see https://finufft.readthedocs.io/en/latest/opts.html
Outputs:
    plan            finufft_plan struct
"""
function finufft_makeplan(type::Integer,
                          n_modes_or_dim::Union{Array{BIGINT},Integer},
                          iflag::Integer,
                          ntrans::Integer,
                          eps::Real;
                          dtype=Float64,
                          kwargs...)
# see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
#   one can also use Array/Vector for cvoid pointer, Array and Ref both work
#   plan_p = Array{finufft_plan_c}(undef,1)
    plan_p = Ref{finufft_plan_c}()

    opts = finufft_default_opts(dtype)
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

    if dtype==Float64
        tol = Float64(eps)
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
                      type,dim,n_modes,iflag,ntrans,tol,plan_p,opts
                      )

    else
        tol = Float32(eps)
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
                      type,dim,n_modes,iflag,ntrans,tol,plan_p,opts
                      )
    end
    check_ret(ret)

    ms = n_modes[1]
    mt = n_modes[2]
    mu = n_modes[3]
    plan = finufft_plan{dtype}(type,ntrans,dim,ms,mt,mu,0,0,plan_p[])
    return plan
end

"""
process nonuniform points for general FINUFFT transform(s).

 With a finufft_plan struct, setpts brings in nonuniform point
 coordinates (xj,yj,zj), and additionally in the type 3 case, nonuniform
 frequency target points (s,t,u). Empty arrays may be passed in the case of
 unused dimensions. For all types, sorting is done to internally store a
 reindexing of points, and for type 3 the spreading and FFTs are planned.
 The nonuniform points may be used for multiple transforms.

Inputs:
    plan   the finufft_plan guru plan struct for one/many general nonuniform FFTs
    xj     vector of x-coords of all nonuniform points
    yj     empty (if dim<2), or vector of y-coords of all nonuniform points
    zj     empty (if dim<3), or vector of z-coords of all nonuniform points
    s      vector of x-coords of all nonuniform frequency targets
    t      empty (if dim<2), or vector of y-coords of all frequency targets
    u      empty (if dim<3), or vector of z-coords of all frequency targets

"""
function finufft_setpts(plan::finufft_plan{T},
                        xj::Array{T},
                        yj::Array{T}=T[],
                        zj::Array{T}=T[],
                        s::Array{T}=T[],
                        t::Array{T}=T[],
                        u::Array{T}=T[]) where T <: finufftReal

    (M, N) = valid_setpts(plan.type, plan.dim, xj, yj, zj, s, t, u)

    plan.nj = M
    plan.nk = N

    if T==Float64
        T_real = Array{Float64}
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
                     plan.plan_ptr,M,T_real(xj),T_real(yj),T_real(zj),N,T_real(s),T_real(t),T_real(u)
                     )
    else
        T_real = Array{Float32}
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
                     plan.plan_ptr,M,T_real(xj),T_real(yj),T_real(zj),N,T_real(s),T_real(t),T_real(u)
                     )
    end

    check_ret(ret)
    return ret
end

"""
execute single or many-vector FINUFFT transforms in a plan.

  output = finufft_exec(plan, input)

  For plan a previously created finufft_plan object also containing all
  needed nonuniform point coordinates, do a single (or if ntrans>1 in the
  plan stage, multiple) NUFFT transform(s), with the strengths or Fourier
  coefficient inputs vector(s) from data_in. The result of the transform(s)
  is returned as a (possibly multidimensional) array.

 Inputs:
     plan     finufft_plan struct
     input    strengths (types 1 or 3) or Fourier coefficients (type 2)
              vector, matrix, or array of appropriate size. For type 1 and 3,
              this is either a length-M vector (where M is the length of xj),
              or an (M,ntrans) matrix when ntrans>1. For type 2, in 1D this is
              length-ms, in 2D size (ms,mt), or in 3D size (ms,mt,mu), or
              each of these with an extra last dimension ntrans if ntrans>1.
 Outputs:
     output   vector of output strengths at targets (types 2 or 3), or array
              of Fourier coefficients (type 1), or, if ntrans>1, a stack of
              such vectors or arrays, of appropriate size.
              Specifically, if ntrans=1, for type 1, in 1D
              this is a length-ms column vector, in 2D a matrix of size
              (ms,mt), or in 3D an array of size (ms,mt,mu); for types 2 and 3
              it is a column vector of length M (the length of xj in type 2),
              or nk (the length of s in type 3). If ntrans>1 its is a stack
              of such objects, ie, it has an extra last dimension ntrans.

"""
function finufft_exec(plan::finufft_plan{T},
                      input::Array{Complex{T}}) where T <: finufftReal
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

"""
To deallocate (delete) a nonuniform FFT plan, use delete(plan)
This deallocates all stored FFTW plans, nonuniform point sorting arrays,
kernel Fourier transforms arrays, etc.
"""
function finufft_destroy(plan::finufft_plan{T}) where T <: finufftReal
    if T==Float64
        ret = ccall( (:finufft_destroy, libfinufft),
                     Cint,
                     (finufft_plan_c,),
                     plan.plan_ptr
                     )
    else
        ret = ccall( (:finufftf_destroy, libfinufft),
                     Cint,
                     (finufft_plan_c,),
                     plan.plan_ptr
                     )
    end
    check_ret(ret)
    return ret
end

"""
Execute single or many-vector FINUFFT transforms in a plan. Output stored in output.
"""
function finufft_exec!(plan::finufft_plan{T},
                      input::Array{Complex{T}},
                      output::Array{Complex{T}}) where T <: finufftReal
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
        if T==Float64
            ret = ccall( (:finufft_execute, libfinufft),
                         Cint,
                         (finufft_plan_c,
                          Ref{ComplexF64},
                          Ref{ComplexF64}),
                         plan.plan_ptr,input,output
                         )
        else
            ret = ccall( (:finufftf_execute, libfinufft),
                         Cint,
                         (finufft_plan_c,
                          Ref{ComplexF32},
                          Ref{ComplexF32}),
                         plan.plan_ptr,input,output
                         )
        end
    elseif type==2
        nj = plan.nj
        if ntrans==1
            @assert size(output)==(nj,ntrans) || size(output)==(nj,)
        else
            @assert size(output)==(nj,ntrans)
        end
        if T==Float64
            ret = ccall( (:finufft_execute, libfinufft),
                         Cint,
                         (finufft_plan_c,
                          Ref{ComplexF64},
                          Ref{ComplexF64}),
                         plan.plan_ptr,output,input
                         )
        else
            ret = ccall( (:finufftf_execute, libfinufft),
                         Cint,
                         (finufft_plan_c,
                          Ref{ComplexF32},
                          Ref{ComplexF32}),
                         plan.plan_ptr,output,input
                         )
        end
    elseif type==3
        nk = plan.nk
        if ntrans==1
            @assert size(output)==(nk,ntrans) || size(output)==(nk,)
        else
            @assert size(output)==(nk,ntrans)
        end
        if T==Float64
            ret = ccall( (:finufft_execute, libfinufft),
                         Cint,
                         (finufft_plan_c,
                          Ref{ComplexF64},
                          Ref{ComplexF64}),
                         plan.plan_ptr,input,output
                         )
        else
            ret = ccall( (:finufftf_execute, libfinufft),
                         Cint,
                         (finufft_plan_c,
                          Ref{ComplexF32},
                          Ref{ComplexF32}),
                         plan.plan_ptr,input,output
                         )
        end
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
end
