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

function finufft_makeplan(type::Integer,
                          n_modes_or_dim::Union{Array{BIGINT},Integer},
                          iflag::Integer,
                          ntrans::Integer,
                          eps::AbstractFloat;
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

function finufft_setpts(plan::finufft_plan{T},
                        xj::Array{T},
                        yj::Array{T}=T[],
                        zj::Array{T}=T[],
                        s::Array{T}=T[],
                        t::Array{T}=T[],
                        u::Array{T}=T[]) where T <: fftwReal

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

function finufft_destroy(plan::finufft_plan{T}) where T <: fftwReal
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

function finufft_exec!(plan::finufft_plan{T},
                      input::Array{Complex{T}},
                      output::Array{Complex{T}}) where T <: fftwReal
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
