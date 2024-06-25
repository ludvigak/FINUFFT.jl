"""
    p = cufinufft_default_opts()

Return a [`FINUFFT.cufinufft_opts`](@ref) struct with the default settings.
"""
function cufinufft_default_opts()
    check_cuda()
    opts = cufinufft_opts()
    ccall( (:cufinufft_default_opts, libcufinufft),
           Cvoid,
           (Ref{cufinufft_opts},),
           opts
           )
    return opts
end

"""
    cufinufft_makeplan(type::Integer,
                       n_modes_or_dim::Union{Array{Int64},Integer},
                       iflag::Integer,
                       ntrans::Integer,
                       eps::Real;
                       dtype=Float64,
                       kwargs...) -> plan::cufinufft_plan{dtype}

Create a `cufinufft_plan` object. See `finufft_makeplan` for arguments.

 - `kwargs` (optional): Options set in [`FINUFFT.cufinufft_opts`](@ref).

"""
function cufinufft_makeplan(type::Integer,
                            n_modes_or_dim::Union{Array{Int64},Integer},
                            iflag::Integer,
                            ntrans::Integer,
                            eps::Real;
                            dtype=Float64,
                            kwargs...)
    _cufinufft_makeplan(dtype, type, n_modes_or_dim, iflag, ntrans, eps; kwargs...)
end

"""
    _cufinufft_makeplan

Type-stable internal version of cufinufft_makeplan
"""
function _cufinufft_makeplan(::Type{dtype},
                            type::Integer,
                            n_modes_or_dim::Union{Array{Int64},Integer},
                            iflag::Integer,
                            ntrans::Integer,
                            eps::Real;
                            kwargs...) where {dtype}
    check_cuda()
# see https://stackoverflow.com/questions/40140699/the-proper-way-to-declare-c-void-pointers-in-julia for how to declare c-void pointers in julia
#   one can also use Array/Vector for cvoid pointer, Array and Ref both work
    plan_ptr = Ref{cufinufft_plan_c}()

    opts = cufinufft_default_opts()
    setkwopts!(opts;kwargs...)

    n_modes = ones(Int64,3)
    if type==3
        throw("Type 3 not implemented yet")
        @assert ndims(n_modes_or_dim) == 0
        dim = n_modes_or_dim
    else
        @assert length(n_modes_or_dim)<=3 && length(n_modes_or_dim)>=1
        dim = length(n_modes_or_dim)
        n_modes[1:dim] .= n_modes_or_dim
    end
    
    if dtype==Float64
        tol = Float64(eps)
        ret = ccall( (:cufinufft_makeplan, libcufinufft),
                      Cint,
                      (Cint,
                       Cint,
                       Ref{Int64},
                       Cint,
                       Cint,
                       Cdouble,
                       Ptr{cufinufft_plan_c},
                       Ref{cufinufft_opts}),
                      type,dim,n_modes,iflag,ntrans,tol,plan_ptr,opts
                      )

    else
        tol = Float32(eps)
        ret = ccall( (:cufinufftf_makeplan, libcufinufft),
                      Cint,
                      (Cint,
                       Cint,
                       Ref{Int64},
                       Cint,
                       Cint,
                       Cfloat,
                       Ptr{cufinufft_plan_c},
                       Ref{cufinufft_opts}),
                      type,dim,n_modes,iflag,ntrans,tol,plan_ptr,opts
                      )
    end
    check_ret(ret)

    ms = n_modes[1]
    mt = n_modes[2]
    mu = n_modes[3]
    plan = cufinufft_plan{dtype}(type,ntrans,dim,ms,mt,mu,0,0,plan_ptr[])
    return plan
end

"""
    cufinufft_destroy!(plan::cufinufft_plan)

Destroy a `cufinufft_plan` object, deallocating all memory used.
"""
function cufinufft_destroy!(plan::cufinufft_plan{T}) where T <: finufftReal
    check_cuda()
    # Remove references to input arrays
    plan._x_d = T[]
    plan._y_d = T[]
    plan._z_d = T[]
    plan._s_d = T[]
    plan._t_d = T[]
    plan._u_d = T[]
    if plan.plan_ptr!=C_NULL # Extra safety, just as in guru.jl
        if T==Float64
            ret = ccall( (:cufinufft_destroy, libcufinufft),
                         Cint,
                         (cufinufft_plan_c,),
                         plan.plan_ptr
                         )
        else
            ret = ccall( (:cufinufftf_destroy, libcufinufft),
                         Cint,
                         (cufinufft_plan_c,),
                         plan.plan_ptr
                         )
        end
        plan.plan_ptr = C_NULL       # signifies destroyed
        check_ret(ret)
    end
end

"""
    cufinufft_setpts!(plan, xj [, yj[, zj[, s[, t[, u]]]]])

Input nonuniform points. See `finufft_setpts!` for arguments.

Points can be either `CUDA.CuArray`'s on device or `Array`'s on host. The latter will be automatically copied to device before being passed to cuFINUFFT.
"""
function cufinufft_setpts!(plan::cufinufft_plan{T},
                           x::Array{T},
                           y::Array{T}=T[],
                           z::Array{T}=T[],
                           s::Array{T}=T[],
                           t::Array{T}=T[],
                           u::Array{T}=T[]) where T <: finufftReal
    # If called with host memory, first copy to device memory
    cufinufft_setpts!(plan,
                      CuArray(x), CuArray(y), CuArray(z),
                      CuArray(s), CuArray(t), CuArray(u))
end

function cufinufft_setpts!(plan::cufinufft_plan{T},
                           x_d::CuArray{T},
                           y_d::CuArray{T}=plan._y_d,
                           z_d::CuArray{T}=plan._z_d,
                           s_d::CuArray{T}=plan._s_d,
                           t_d::CuArray{T}=plan._t_d,
                           u_d::CuArray{T}=plan._u_d) where T <: finufftReal
    check_cuda()
    (M, N) = valid_setpts(plan.type, plan.dim, x_d, y_d, z_d, s_d, t_d, u_d)

    plan.nj = M
    plan.nk = N

    # Store references to input arrays in plan struct.
    # This is important, since Julia garbage collection
    # will not know about the C library keeping references
    # to the input arrays.
    plan._x_d = vec(x_d)
    plan._y_d = vec(y_d)
    plan._z_d = vec(z_d)
    plan._s_d = vec(s_d)
    plan._t_d = vec(t_d)
    plan._u_d = vec(u_d)

    if T==Float64
        ret = ccall( (:cufinufft_setpts, libcufinufft),
                     Cint,
                     (cufinufft_plan_c,
                      Int64,
                      CuRef{Cdouble},
                      CuRef{Cdouble},
                      CuRef{Cdouble},
                      Int64,
                      CuRef{Cdouble},
                      CuRef{Cdouble},
                      CuRef{Cdouble}),
                     plan.plan_ptr, M, x_d, y_d, z_d, N, s_d, t_d, u_d
                     )
    else
        ret = ccall( (:cufinufftf_setpts, libcufinufft),
                     Cint,
                     (cufinufft_plan_c,
                      Int64,
                      CuRef{Cfloat},
                      CuRef{Cfloat},
                      CuRef{Cfloat},
                      Int64,
                      CuRef{Cfloat},
                      CuRef{Cfloat},
                      CuRef{Cfloat}),
                     plan.plan_ptr, M, x_d, y_d, z_d, N, s_d, t_d, u_d
                     )
    end

    check_ret(ret)
    return ret
end
"""
    cufinufft_exec(plan::cufinufft_plan{T}, 
                   input :: Array{Complex{T}} or CUDA.CuArray{Complex{T}}
                   ) -> Array{Complex{T}} or CUDA.CuArray{Complex{T}}
                   where T :: Float32 or Float64

Execute cuFINFFT plan and return output in a newly allocated array.

`output` type will match that of `input`:
- If `input` is `CUDA.CuArray` on device, then `output` is allocated on device.
- If `input` is `Array` on host, then it is copied to device before computation and `output` is copied to host after computation.
"""
function cufinufft_exec(plan::cufinufft_plan{T},
                        input_h::Array{Complex{T}}
                        ) :: Array{Complex{T}} where T <: finufftReal
    # If called with host memory, return host memory
    input_d = CuArray(input_h)
    output_d = cufinufft_exec(plan, input_d)
    output_h = Array(output_d)
    return Array(output_h)
end

function cufinufft_exec(plan::cufinufft_plan{T},
                        input::CuArray{Complex{T}}) :: CuArray{Complex{T}} where T <: finufftReal
    ret = 0
    type = plan.type
    ntrans = plan.ntrans
    dim = plan.dim
    n_modes = Array{Int64}(undef,3)
    n_modes[1] = plan.ms
    n_modes[2] = plan.mt
    n_modes[3] = plan.mu
    if type==1
        if dim==1
            output = CuArray{Complex{T}}(undef,n_modes[1],ntrans)
        elseif dim==2
            output = CuArray{Complex{T}}(undef,n_modes[1],n_modes[2],ntrans)
        elseif dim==3
            output = CuArray{Complex{T}}(undef,n_modes[1],n_modes[2],n_modes[3],ntrans)
        else
            ret = ERR_DIM_NOTVALID
        end
    elseif type==2
        nj = plan.nj
        output = CuArray{Complex{T}}(undef,nj,ntrans)
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
    cufinufft_exec!(plan,input,output)
    return output
end

"""
    cufinufft_exec!(plan::cufinufft_plan{T},
                    input::CUDA.CuArray{Complex{T}},
                    output::CUDA.CuArray{Complex{T}}
                    ) 
                    where T :: Float32 or Float64

Execute cuFINUFFT transform(s) with preallocated arrays on device.
"""
function cufinufft_exec!(plan::cufinufft_plan{T},
                         input::CuArray{Complex{T}},
                         output::CuArray{Complex{T}}) where T <: finufftReal
    check_cuda()
    type = plan.type
    ntrans = plan.ntrans
    dim = plan.dim
    n_modes = (plan.ms, plan.mt, plan.mu)
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
            ret = ccall( (:cufinufft_execute, libcufinufft),
                         Cint,
                         (cufinufft_plan_c,
                          CuRef{ComplexF64},
                          CuRef{ComplexF64}),
                         plan.plan_ptr,input,output
                         )
        else
            ret = ccall( (:cufinufftf_execute, libcufinufft),
                         Cint,
                         (cufinufft_plan_c,
                          CuRef{ComplexF32},
                          CuRef{ComplexF32}),
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
            ret = ccall( (:cufinufft_execute, libcufinufft),
                         Cint,
                         (cufinufft_plan_c,
                          CuRef{ComplexF64},
                          CuRef{ComplexF64}),
                         plan.plan_ptr,output,input
                         )
        else
            ret = ccall( (:cufinufftf_execute, libcufinufft),
                         Cint,
                         (cufinufft_plan_c,
                          CuRef{ComplexF32},
                          CuRef{ComplexF32}),
                         plan.plan_ptr,output,input
                         )
        end
    else
        ret = ERR_TYPE_NOTVALID
    end
    check_ret(ret)
end


