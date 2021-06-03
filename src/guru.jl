### Guru Interfaces
finufft_plan{T} = Ptr{T} where T <: Union{Float32,Float64}

function finufft_makeplan(type::Integer,
                          dim::Integer,
                          n_modes::T,
                          iflag::Integer,
                          ntrans::Integer,
                          eps::Float64,
                          opts::nufft_opts=finufft_default_opts()) where T <: Union{Array{BIGINT}, NTuple{3,BIGINT}}

    # see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
    plan_p = Ref{finufft_plan{Float64}}()
    
    ret = ccall( (:finufft_makeplan, libfinufft),
            Cint,
            (Cint,
            Cint,
            Ref{BIGINT},
            Cint,
            Cint,
            Cdouble,
            Ptr{finufft_plan{Cdouble}},
            Ref{nufft_opts}),
            type,dim,n_modes,iflag,ntrans,eps,plan_p,opts
            )
    check_ret(ret)
    
    plan = plan_p[]
    return plan
end
function finufft_makeplan(type::Integer,
    dim::Integer,
    n_modes::T,
    iflag::Integer,
    ntrans::Integer,
    eps::Float32,
    opts::nufft_opts=finufft_default_opts()) where T <: Union{Array{BIGINT}, NTuple{3,BIGINT}}
    
    # see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
    plan_p = Ref{finufft_plan{Float32}}()

    ret = ccall( (:finufftf_makeplan, libfinufft),
            Cint,
            (Cint,
            Cint,
            Ref{BIGINT},
            Cint,
            Cint,
            Cfloat,
            Ptr{finufft_plan{Cfloat}},
            Ref{nufft_opts}),
            type,dim,n_modes,iflag,ntrans,eps,plan_p,opts
            )
    check_ret(ret)
    
    plan = plan_p[]
    return plan
end


function finufft_setpts(plan::finufft_plan{Float64},
                        xj::Array{Float64},
                        yj::Array{Float64}=Float64[],
                        zj::Array{Float64}=Float64[],
                        s::Array{Float64}=Float64[],
                        t::Array{Float64}=Float64[],
                        u::Array{Float64}=Float64[])

    nj = length(xj)
    nk = length(s)

    ret = ccall( (:finufft_setpts, libfinufft),
                Cint,
                (finufft_plan{Cdouble},
                BIGINT,
                Ref{Cdouble},
                Ref{Cdouble},
                Ref{Cdouble},
                BIGINT,
                Ref{Cdouble},
                Ref{Cdouble},
                Ref{Cdouble}),
                plan,nj,xj,yj,zj,nk,s,t,u
                )

    check_ret(ret)
end
function finufft_setpts(plan::finufft_plan{Float32},
    xj::Array{Float32},
    yj::Array{Float32}=Float32[],
    zj::Array{Float32}=Float32[],
    s::Array{Float32}=Float32[],
    t::Array{Float32}=Float32[],
    u::Array{Float32}=Float32[])

    nj = length(xj)
    nk = length(s)
    
    ret = ccall( (:finufftf_setpts, libfinufft),
                Cint,
                (finufft_plan{Cfloat},
                BIGINT,
                Ref{Cfloat},
                Ref{Cfloat},
                Ref{Cfloat},
                BIGINT,
                Ref{Cfloat},
                Ref{Cfloat},
                Ref{Cfloat}),
                plan,nj,xj,yj,zj,nk,s,t,u
                )

    check_ret(ret)
end


function finufft_exec(plan::finufft_plan{Float64}, cj::Array{ComplexF64}, fk::Array{ComplexF64})

    ret = ccall( (:finufft_execute, libfinufft),
                Cint,
                (finufft_plan{Cdouble},
                Ref{ComplexF64},
                Ref{ComplexF64}),
                plan,cj,fk
                )

    check_ret(ret)
end
function finufft_exec(plan::finufft_plan{Float32}, cj::Array{ComplexF32}, fk::Array{ComplexF32})
        
    ret = ccall( (:finufftf_execute, libfinufft),
                Cint,
                (finufft_plan{Cfloat},
                Ref{ComplexF32},
                Ref{ComplexF32}),
                plan,cj,fk
                )

    check_ret(ret)
end

function finufft_destroy(plan::finufft_plan{Float64})
    ret = ccall( (:finufft_destroy, libfinufft),
                Cint,
                (finufft_plan{Cdouble},),
                plan
                )
    
    check_ret(ret)
end
function finufft_destroy(plan::finufft_plan{Float32})
    ret = ccall( (:finufftf_destroy, libfinufft),
            Cint,
            (finufft_plan{Cfloat},),
            plan
            )

    check_ret(ret)
end
