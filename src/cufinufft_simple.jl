### Simple interfaces, No allocation
#
# These methods overload the regular interface for CuArray inputs

## 1D

"""
    nufft1d1!(xj      :: CuArray{Float64} or CuArray{Float32},
              cj      :: CuArray{ComplexF64} or CuArray{ComplexF32},
              iflag   :: Integer,
              eps     :: Real,
              fk      :: CuArray{ComplexF64} or CuArray{ComplexF32};
              kwargs...
            )

CUDA version.
"""
function nufft1d1!(xj      :: CuArray{T},
                   cj      :: CuArray{Complex{T}},
                   iflag   :: Integer,
                   eps     :: Real,
                   fk      :: CuArray{Complex{T}};
                   kwargs...) where T <: finufftReal
    valid_setpts(1,1,xj)
    ntrans = valid_ntr(xj,cj)
    (ms, ntrans_fk) = get_nmodes_from_fk(1,fk)

    checkkwdtype(T; kwargs...)
    plan = _cufinufft_makeplan(T,1,[ms;],iflag,ntrans,eps;kwargs...)
    cufinufft_setpts!(plan,xj)
    cufinufft_exec!(plan,cj,fk)
    cufinufft_destroy!(plan)
end


"""
    nufft1d2!(xj      :: CuArray{Float64} or CuArray{Float32},
              cj      :: CuArray{ComplexF64} or CuArray{ComplexF32},
              iflag   :: Integer,
              eps     :: Real,
              fk      :: CuArray{ComplexF64} or CuArray{ComplexF32};
              kwargs...
            )

CUDA version.
"""
function nufft1d2!(xj      :: CuArray{T},
                   cj      :: CuArray{Complex{T}},
                   iflag   :: Integer,
                   eps     :: Real,
                   fk      :: CuArray{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,1,xj)
    (ms, ntrans) = get_nmodes_from_fk(1,fk)

    checkkwdtype(T; kwargs...)
    plan = _cufinufft_makeplan(T,2,[ms;],iflag,ntrans,eps;kwargs...)
    cufinufft_setpts!(plan,xj)
    cufinufft_exec!(plan,fk,cj)
    cufinufft_destroy!(plan)
end

## 2D

"""
    nufft2d1!(xj      :: CuArray{Float64} or CuArray{Float32},
              yj      :: CuArray{Float64} or CuArray{Float32},
              cj      :: CuArray{ComplexF64} or CuArray{ComplexF32},
              iflag   :: Integer,
              eps     :: Real,
              fk      :: CuArray{ComplexF64} or CuArray{ComplexF32};
              kwargs...
            )

CUDA version.
"""
function nufft2d1!(xj      :: CuArray{T},
                   yj      :: CuArray{T},
                   cj      :: CuArray{Complex{T}},
                   iflag   :: Integer,
                   eps     :: Real,
                   fk      :: CuArray{Complex{T}};
                   kwargs...) where T <: finufftReal
    valid_setpts(1,2,xj,yj)
    ntrans = valid_ntr(xj,cj)
    (ms, mt, ntrans_fk) = get_nmodes_from_fk(2,fk)
    @assert ntrans==ntrans_fk

    checkkwdtype(T; kwargs...)
    plan = _cufinufft_makeplan(T,1,[ms;mt],iflag,ntrans,eps;kwargs...)
    cufinufft_setpts!(plan,xj,yj)
    cufinufft_exec!(plan,cj,fk)
    cufinufft_destroy!(plan)
end


"""
    nufft2d2!(xj      :: CuArray{Float64} or CuArray{Float32},
              yj      :: CuArray{Float64} or CuArray{Float32},
              cj      :: CuArray{ComplexF64} or CuArray{ComplexF32},
              iflag   :: Integer,
              eps     :: Real,
              fk      :: CuArray{ComplexF64} or CuArray{ComplexF32};
              kwargs...
            )

CUDA version.
"""
function nufft2d2!(xj      :: CuArray{T},
                   yj      :: CuArray{T},
                   cj      :: CuArray{Complex{T}},
                   iflag   :: Integer,
                   eps     :: Real,
                   fk      :: CuArray{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(1,2,xj,yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2,fk)

    checkkwdtype(T; kwargs...)
    plan = _cufinufft_makeplan(T,2,[ms;mt],iflag,ntrans,eps;kwargs...)
    cufinufft_setpts!(plan,xj,yj)
    cufinufft_exec!(plan,fk,cj)
    cufinufft_destroy!(plan)
end

## 3D

"""
    nufft3d1!(xj      :: CuArray{Float64} or CuArray{Float32},
              yj      :: CuArray{Float64} or CuArray{Float32},
              zj      :: CuArray{Float64} or CuArray{Float32},
              cj      :: CuArray{ComplexF64} or CuArray{ComplexF32},
              iflag   :: Integer,
              eps     :: Real,
              fk      :: CuArray{ComplexF64} or CuArray{ComplexF32};
              kwargs...
            )

CUDA version.
"""
function nufft3d1!(xj      :: CuArray{T},
                   yj      :: CuArray{T},
                   zj      :: CuArray{T},
                   cj      :: CuArray{Complex{T}},
                   iflag   :: Integer,
                   eps     :: Real,
                   fk      :: CuArray{Complex{T}};
                   kwargs...) where T <: finufftReal
    valid_setpts(1,3,xj,yj,zj)
    ntrans = valid_ntr(xj,cj)
    (ms, mt, mu, ntrans_fk) = get_nmodes_from_fk(3,fk)
    @assert ntrans == ntrans_fk

    checkkwdtype(T; kwargs...)
    plan = _cufinufft_makeplan(T,1,[ms;mt;mu],iflag,ntrans,eps;kwargs...)
    cufinufft_setpts!(plan,xj,yj,zj)
    cufinufft_exec!(plan,cj,fk)
    cufinufft_destroy!(plan)
end

"""
    nufft3d2!(xj      :: CuArray{Float64} or CuArray{Float32},
              yj      :: CuArray{Float64} or CuArray{Float32},
              zj      :: CuArray{Float64} or CuArray{Float32},
              cj      :: CuArray{ComplexF64} or CuArray{ComplexF32},
              iflag   :: Integer,
              eps     :: Real,
              fk      :: CuArray{ComplexF64} or CuArray{ComplexF32};
              kwargs...
            )

CUDA version.
"""
function nufft3d2!(xj      :: CuArray{T},
                   yj      :: CuArray{T},
                   zj      :: CuArray{T},
                   cj      :: CuArray{Complex{T}},
                   iflag   :: Integer,
                   eps     :: Real,
                   fk      :: CuArray{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,3,xj,yj,zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3,fk)

    checkkwdtype(T; kwargs...)
    plan = _cufinufft_makeplan(T,2,[ms;mt;mu],iflag,ntrans,eps;kwargs...)
    cufinufft_setpts!(plan,xj,yj,zj)
    cufinufft_exec!(plan,fk,cj)
    cufinufft_destroy!(plan)
end
