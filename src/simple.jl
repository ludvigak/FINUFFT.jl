### Simple Interfaces (allocate output)
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
function nufft1d1(xj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer,
                  eps::T,
                  ms::Integer;
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    valid_setpts(1, 1, xj)
    ntrans = valid_ntr(xj, cj)
    fk = Array{Complex{dtype}}(undef, ms, ntrans)
    checkkwdtype(dtype; kwargs...)
    nufft1d1!(xj, cj, iflag, tol, fk; kwargs...)
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
function nufft2d1(xj::Array{T},
                  yj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer, 
                  eps::T,
                  ms::Integer,
                  mt::Integer;
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    valid_setpts(1, 2, xj, yj)
    ntrans = valid_ntr(xj, cj)
    fk = Array{Complex{T}}(undef, ms, mt, ntrans)
    checkkwdtype(T; kwargs...)
    nufft2d1!(xj, yj, cj, iflag, tol, fk; kwargs...)
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
function nufft3d1(xj::Array{T},
                  yj::Array{T},
                  zj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer, 
                  eps::T,
                  ms::Integer,
                  mt::Integer,
                  mu::Integer;
                  dtype::DataType=T,
                  kwargs...)  where T <: fftwReal
    tol = dtype(eps)
    valid_setpts(1, 3, xj, yj, zj)
    ntrans = valid_ntr(xj, cj)
    fk = Array{Complex{T}}(undef, ms, mt, mu, ntrans)
    checkkwdtype(T; kwargs...)
    nufft3d1!(xj, yj, zj, cj, iflag, tol, fk; kwargs...)
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
function nufft1d2(xj::Array{T},
                  iflag::Integer,
                  eps::T,
                  fk::Array{Complex{T}};
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(2, 1, xj)
    (ms, ntrans) = get_nmodes_from_fk(1, fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    checkkwdtype(T; kwargs...)
    nufft1d2!(xj, cj, iflag, tol, fk; kwargs...)
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
function nufft2d2(xj::Array{T},
                  yj::Array{T},
                  iflag::Integer,
                  eps::T,
                  fk::Array{Complex{T}};
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(2, 2, xj, yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2, fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    checkkwdtype(T; kwargs...)
    nufft2d2!(xj, yj, cj, iflag, tol, fk; kwargs...)
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
function nufft3d2(xj::Array{T},
                  yj::Array{T},
                  zj::Array{T},
                  iflag::Integer, 
                  eps::T,
                  fk::Array{Complex{T}};
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(2, 3, xj, yj, zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3, fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    checkkwdtype(T; kwargs...)
    nufft3d2!(xj, yj, zj, cj, iflag, tol, fk; kwargs...)
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
function nufft1d3(xj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer, 
                  eps::T,
                  sk::Array{T};
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(3, 1, xj, T[], T[], sk)
    ntrans = valid_ntr(xj, cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    checkkwdtype(T; kwargs...)
    nufft1d3!(xj, cj, iflag, tol, sk, fk; kwargs...)
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
function nufft2d3(xj::Array{T},
                  yj::Array{T}, 
                  cj::Array{Complex{T}}, 
                  iflag::Integer, 
                  eps::T,
                  sk::Array{T},
                  tk::Array{T};
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(3, 2, xj, yj, T[], sk, tk)
    ntrans = valid_ntr(xj, cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    checkkwdtype(T; kwargs...)
    nufft2d3!(xj, yj, cj, iflag, tol, sk, tk, fk; kwargs...)
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
function nufft3d3(xj::Array{T},
                  yj::Array{T},
                  zj::Array{T},                   
                  cj::Array{Complex{T}}, 
                  iflag::Integer, 
                  eps::T,
                  sk::Array{T},
                  tk::Array{T},
                  uk::Array{T};
                  dtype::DataType=T,
                  kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(3, 3, xj, yj, zj, sk, tk, uk)
    ntrans = valid_ntr(xj, cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    checkkwdtype(T; kwargs...)
    nufft3d3!(xj, yj, zj, cj, iflag, tol, sk, tk, uk, fk; kwargs...)
    return fk
end


### Direct interfaces (No allocation)
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
function nufft1d1!(xj::Array{T},
                   cj::Array{Complex{T}},
                   iflag::Integer, 
                   eps::T,
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    valid_setpts(1, 1, xj)
    ntrans = valid_ntr(xj, cj)
    (ms, ntrans_fk) = get_nmodes_from_fk(1, fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(1, [ms;], iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj)
    finufft_exec!(plan, cj, fk)
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
function nufft1d2!(xj::Array{T},
                   cj::Array{Complex{T}},
                   iflag::Integer, 
                   eps::T,
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(2, 1, xj)
    (ms, ntrans) = get_nmodes_from_fk(1, fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(2, [ms;], iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj)
    finufft_exec!(plan, fk, cj)
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
function nufft1d3!(xj::Array{T},
                   cj::Array{Complex{T}},
                   iflag::Integer, 
                   eps::T,
                   sk::Array{T},
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(3, 1, xj, T[], T[], sk)
    ntrans = valid_ntr(xj, cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(3, 1, iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, T[], T[], sk)
    finufft_exec!(plan, cj, fk)
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
function nufft2d1!(xj::Array{T}, 
                   yj::Array{T}, 
                   cj::Array{Complex{T}}, 
                   iflag::Integer, 
                   eps::T,
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    valid_setpts(1, 2, xj, yj)
    ntrans = valid_ntr(xj, cj)
    (ms, mt, ntrans_fk) = get_nmodes_from_fk(2, fk)
    @assert ntrans == ntrans_fk

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(1, [ms;mt], iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, yj)
    finufft_exec!(plan, cj, fk)
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
function nufft2d2!(xj::Array{T}, 
                   yj::Array{T}, 
                   cj::Array{Complex{T}}, 
                   iflag::Integer, 
                   eps::T,
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(1, 2, xj, yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2, fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(2, [ms;mt], iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, yj)
    finufft_exec!(plan, fk, cj)
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
function nufft2d3!(xj::Array{T}, 
                   yj::Array{T},
                   cj::Array{Complex{T}}, 
                   iflag::Integer, 
                   eps::T,
                   sk::Array{T},
                   tk::Array{T},
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(3, 2, xj, yj, T[], sk, tk)
    ntrans = valid_ntr(xj, cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(3, 2, iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, yj, T[], sk, tk)
    finufft_exec!(plan, cj, fk)
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
function nufft3d1!(xj::Array{T}, 
                   yj::Array{T}, 
                   zj::Array{T}, 
                   cj::Array{Complex{T}}, 
                   iflag::Integer, 
                   eps::T,
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    valid_setpts(1, 3, xj, yj, zj)
    ntrans = valid_ntr(xj, cj)
    (ms, mt, mu, ntrans_fk) = get_nmodes_from_fk(3, fk)
    @assert ntrans == ntrans_fk

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(1, [ms;mt;mu], iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, yj, zj)
    finufft_exec!(plan, cj, fk)
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
function nufft3d2!(xj::Array{T}, 
                   yj::Array{T},
                   zj::Array{T},                    
                   cj::Array{Complex{T}}, 
                   iflag::Integer, 
                   eps::T,
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(2, 3, xj, yj, zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3, fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(2, [ms;mt;mu], iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, yj, zj)
    finufft_exec!(plan, fk, cj)
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
function nufft3d3!(xj::Array{T}, 
                   yj::Array{T},
                   zj::Array{T},                   
                   cj::Array{Complex{T}}, 
                   iflag::Integer, 
                   eps::T,
                   sk::Array{T},
                   tk::Array{T},
                   uk::Array{T},
                   fk::Array{Complex{T}};
                   dtype::DataType=T,
                   kwargs...) where T <: fftwReal
    tol = dtype(eps)
    (nj, nk) = valid_setpts(3, 3, xj, yj, zj, sk, tk, uk)
    ntrans = valid_ntr(xj, cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(3, 3, iflag, ntrans, tol; kwargs...)
    finufft_setpts(plan, xj, yj, zj, sk, tk, uk)
    finufft_exec!(plan, cj, fk)
    ret = finufft_destroy(plan)
    check_ret(ret)
end