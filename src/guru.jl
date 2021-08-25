### Guru Interfaces
finufft_plan_c = Ptr{Cvoid}

mutable struct finufft_plan{T}
    type::Cint
    ntrans::Cint
    dim::Cint
    ms::BIGINT
    mt::BIGINT
    mu::BIGINT
    nj::BIGINT
    nk::BIGINT
    plan_ptr::finufft_plan_c
end



for (T, cfun) ∈ ((Float64, (:finufft_makeplan, libfinufft)), (Float32, (:finufftf_makeplan, libfinufft)))
    @eval begin
        function finufft_makeplan(type::Integer,
                            n_modes_or_dim::Union{Array{BIGINT},Integer},
                            iflag::Integer,
                            ntrans::Integer,
                            eps::$T;
                            kwargs...)
        # see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
        #   one can also use Array/Vector for cvoid pointer, Array and Ref both work
            plan_p = Ref{finufft_plan_c}()
        
            opts = finufft_default_opts(nufft_opts{$T}())
            setkwopts!(opts;kwargs...)
        
            n_modes = ones(BIGINT, 3)
            if type == 3
                @assert ndims(n_modes_or_dim) == 0
                dim = n_modes_or_dim
            else
                @assert length(n_modes_or_dim) <= 3 && length(n_modes_or_dim) >= 1
                dim = length(n_modes_or_dim)
                n_modes[1:dim] .= n_modes_or_dim
            end
        
            ret = ccall( $cfun,
                            Cint,
                            (Cint,
                            Cint,
                            Ref{BIGINT},
                            Cint,
                            Cint,
                            $T,
                            Ptr{finufft_plan_c},
                            Ref{nufft_opts}),
                            type,dim,n_modes,iflag,ntrans,eps,plan_p,opts
                            )
            
            check_ret(ret)
        
            ms = n_modes[1]
            mt = n_modes[2]
            mu = n_modes[3]
            plan = finufft_plan{$T}(type, ntrans, dim, ms, mt, mu, 0, 0, plan_p[])
        end
    end
end

for (T, cfun) ∈ ((Float64, (:finufft_setpts, libfinufft)), (Float32, (:finufftf_setpts, libfinufft)))
    @eval begin
        function finufft_setpts(plan::finufft_plan{$T},
                                xj::Array{$T},
                                yj::Array{$T}=$T[],
                                zj::Array{$T}=$T[],
                                s::Array{$T}=$T[],
                                t::Array{$T}=$T[],
                                u::Array{$T}=$T[])

            (M, N) = valid_setpts(plan.type, plan.dim, xj, yj, zj, s, t, u)

            plan.nj = M
            plan.nk = N

            ret = ccall( $cfun,
                            Cint,
                            (finufft_plan_c,
                            BIGINT,
                            Ref{$T},
                            Ref{$T},
                            Ref{$T},
                            BIGINT,
                            Ref{$T},
                            Ref{$T},
                            Ref{$T}),
                            plan.plan_ptr,M,xj,yj,zj,N,s,t,u
                            )
            check_ret(ret)
            return ret
        end
    end
end


function finufft_exec(plan::finufft_plan{T},
                      input::Array{Complex{T}}) where T <: fftwReal
    ret = 0
    type = plan.type
    ntrans = plan.ntrans
    dim = plan.dim
    n_modes = Array{BIGINT}(undef, 3)
    n_modes[1] = plan.ms
    n_modes[2] = plan.mt
    n_modes[3] = plan.mu
    if type == 1
        if dim == 1
            output = Array{Complex{T}}(undef, n_modes[1], ntrans)
        elseif dim == 2
            output = Array{Complex{T}}(undef, n_modes[1], n_modes[2], ntrans)
        elseif dim == 3
            output = Array{Complex{T}}(undef, n_modes[1], n_modes[2], n_modes[3], ntrans)
        else
            ret = ERR_DIM_NOTVALID
        end
    elseif type == 2
        nj = plan.nj
        output = Array{Complex{T}}(undef, nj, ntrans)
    elseif type == 3
        nk = plan.nk
        output = Array{Complex{T}}(undef, nk, ntrans)
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
    finufft_exec!(plan, input, output)
    return output
end

for (T, cfun) ∈ ((Float64, (:finufft_destroy, libfinufft)), (Float32, (:finufftf_destroy, libfinufft)))
    @eval begin
        function finufft_destroy(plan::finufft_plan{$T})
            ret = ccall( $cfun,
                        Cint,
                        (finufft_plan_c,),
                        plan.plan_ptr
                        )
            check_ret(ret)
            return ret
        end
    end
end

for (T, cfun) ∈ ((Float64, (:finufft_execute, libfinufft)), (Float32, (:finufftf_execute, libfinufft)))
    @eval begin
        function finufft_exec!(plan::finufft_plan{$T},
                            input::Array{Complex{$T}},
                            output::Array{Complex{$T}})
            type = plan.type
            ntrans = plan.ntrans
            dim = plan.dim
            n_modes = Array{BIGINT}(undef, 3)
            n_modes[1] = plan.ms
            n_modes[2] = plan.mt
            n_modes[3] = plan.mu
            if type == 1
                if dim == 1
                    if ntrans == 1
                        @assert size(output) == (n_modes[1],) || size(output) == (n_modes[1], ntrans)
                    else
                        @assert size(output) == (n_modes[1], ntrans)
                    end
                elseif dim == 2
                    if ntrans == 1
                        @assert size(output) == (n_modes[1], n_modes[2]) || size(output) == (n_modes[1], n_modes[2], ntrans)
                    else
                        @assert size(output) == (n_modes[1], n_modes[2], ntrans)
                    end
                elseif dim == 3
                    if ntrans == 1
                        @assert size(output) == (n_modes[1], n_modes[2], n_modes[3]) || size(output) == (n_modes[1], n_modes[2], n_modes[3], ntrans)
                    else
                        @assert size(output) == (n_modes[1], n_modes[2], n_modes[3], ntrans)
                    end
                else
                    ret = ERR_DIM_NOTVALID
                    check_ret(ret)
                end
                ret = ccall( $cfun,
                                Cint,
                                (finufft_plan_c,
                                Ref{Complex{$T}},
                                Ref{Complex{$T}}),
                                plan.plan_ptr,input,output
                                )
            elseif type == 2
                nj = plan.nj
                if ntrans == 1
                    @assert size(output) == (nj, ntrans) || size(output) == (nj,)
                else
                    @assert size(output) == (nj, ntrans)
                end
                ret = ccall( $cfun,
                                Cint,
                                (finufft_plan_c,
                                Ref{Complex{$T}},
                                Ref{Complex{$T}}),
                                plan.plan_ptr,output,input
                                )
            elseif type == 3
                nk = plan.nk
                if ntrans == 1
                    @assert size(output) == (nk, ntrans) || size(output) == (nk,)
                else
                    @assert size(output) == (nk, ntrans)
                end
                ret = ccall( $cfun,
                                Cint,
                                (finufft_plan_c,
                                Ref{Complex{$T}},
                                Ref{Complex{$T}}),
                                plan.plan_ptr,input,output
                                )
            else
                ret = ERR_TYPE_NOTVALID
            end
            check_ret(ret)
        end
    end
end