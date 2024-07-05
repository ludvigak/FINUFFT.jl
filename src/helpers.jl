# HELPER routines for guru interface...

### validate sizes of inputs for setpts
function valid_setpts(type::Integer,
                      dim::Integer,
                      x::AbstractArray{T},
                      y::AbstractArray{T}=T[],
                      z::AbstractArray{T}=T[],
                      s::AbstractArray{T}=T[],
                      t::AbstractArray{T}=T[],
                      u::AbstractArray{T}=T[]) where T <: finufftReal
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
function valid_ntr(x::AbstractArray{T},
                   c::AbstractArray{Complex{T}}) where T <: finufftReal
    ntrans = Cint(length(c) / length(x))
    @assert ntrans*length(x) == length(c)
    return ntrans
end

### infer number of modes from fk array
function get_nmodes_from_fk(dim::Integer,
                            fk::AbstractArray{Complex{T}}) where T <: finufftReal
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
function setkwopts!(opts::Union{nufft_opts,cufinufft_opts}; kwargs...)
    dtype = Float64
    for (key, value) in kwargs
        if hasproperty(opts, key::Symbol)
            setproperty!(opts, key, value)
        elseif String(key)=="dtype"
            @assert value <: finufftReal
            dtype = value
        else
            @warn string(typeof(opts)) * " does not have attribute " * String(key)
        end
    end
    return dtype
end

### check kwargs with dtype
function checkkwdtype(dtype::DataType; kwargs...)
    for (key, value) in kwargs
        if String(key)=="dtype"
            @warn "Explicitly passing the dtype argument is discouraged and will be deprecated."
            @assert  value == dtype
        end
    end
end
