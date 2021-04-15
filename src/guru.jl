### Guru Interfaces
finufft_plan{T} = Ptr{T} where T <: Union{Float32,Float64}

function finufft_makeplan(type::Integer,
                          dim::Integer,
                          n_modes::Array{BIGINT},
                          iflag::Integer,
                          ntrans::Integer,
                          eps::Float64,
                          opts::nufft_opts=finufft_default_opts())

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
    n_modes::Array{BIGINT},
    iflag::Integer,
    ntrans::Integer,
    eps::Float32,
    opts::nufft_opts=finufft_default_opts())
    
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
                        xj::StridedArray{Float64},
                        yj::StridedArray{Float64}=Float64[],
                        zj::StridedArray{Float64}=Float64[],
                        s::StridedArray{Float64}=Float64[],
                        t::StridedArray{Float64}=Float64[],
                        u::StridedArray{Float64}=Float64[])

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
    return ret
end
function finufft_setpts(plan::finufft_plan{Float32},
    xj::StridedArray{Float32},
    yj::StridedArray{Float32}=Float32[],
    zj::StridedArray{Float32}=Float32[],
    s::StridedArray{Float32}=Float32[],
    t::StridedArray{Float32}=Float32[],
    u::StridedArray{Float32}=Float32[])

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
    return ret
end


function finufft_exec(plan::finufft_plan{Float64}, cj::StridedArray{ComplexF64}, fk::StridedArray{ComplexF64})

    ret = ccall( (:finufft_execute, libfinufft),
                Cint,
                (finufft_plan{Cdouble},
                Ref{ComplexF64},
                Ref{ComplexF64}),
                plan,cj,fk
                )

    check_ret(ret)
    return ret
end
function finufft_exec(plan::finufft_plan{Float32}, cj::StridedArray{ComplexF32}, fk::StridedArray{ComplexF32})
        
    ret = ccall( (:finufftf_execute, libfinufft),
                Cint,
                (finufft_plan{Cfloat},
                Ref{ComplexF32},
                Ref{ComplexF32}),
                plan,cj,fk
                )

    check_ret(ret)
    return ret
end

function finufft_destroy(plan::finufft_plan{Float64})
    ret = ccall( (:finufft_destroy, libfinufft),
                Cint,
                (finufft_plan{Cdouble},),
                plan
                )
    
    check_ret(ret)
    return ret
end
function finufft_destroy(plan::finufft_plan{Float32})
    ret = ccall( (:finufftf_destroy, libfinufft),
            Cint,
            (finufft_plan{Cfloat},),
            plan
            )

    check_ret(ret)
    return ret
end
