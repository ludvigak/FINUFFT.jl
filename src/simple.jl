### Simple Interfaces (allocate output)
### Double precision
## Type-1

"""
    nufft1d1(xj      :: Array{Float64} or Array{Float32}, 
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Real,
             ms      :: Integer;
             kwargs...
            ) -> Array{ComplexF64} or Array{ComplexF32}

Compute type-1 1D complex nonuniform FFT. 
This computes, to relative precision eps, via a fast algorithm:

              nj
    f(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2
             j=1
 # Inputs
  - `xj`      locations of nonuniform sources on interval [-3pi,3pi), length nj
  - `cj`      length-nj complex vector of source strengths. If length(cj)>nj,
            expects a stack of vectors (eg, a nj*ntrans matrix) each of which is
            transformed with the same source locations.
  - `iflag`   if >=0, uses + sign in exponential, otherwise - sign.
  - `eps`     relative precision requested (generally between 1e-15 and 1e-1)
  - `ms`      number of Fourier modes computed, may be even or odd;
            in either case, mode range is integers lying in [-ms/2, (ms-1)/2]
  - kwargs  (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 # Output
  - size `(ms,)` complex vector of Fourier coefficients f, or, if
            `ntrans>1`, matrix of size `(ms,ntrans)`.

"""
function nufft1d1(xj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer,
                  eps::Real,
                  ms::Integer;
                  kwargs...) where T <: finufftReal
    valid_setpts(1,1,xj)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, ms, ntrans)
    checkkwdtype(T; kwargs...)
    nufft1d1!(xj, cj, iflag, eps, fk; kwargs...)
    return fk
end

"""
    nufft2d1(xj      :: Array{Float64} or Array{Float32}
             yj      :: Array{Float64} or Array{Float32}, 
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Real,
             ms      :: Integer,
             mt      :: Integer;
             kwargs...
            ) -> Array{ComplexF64}

Compute type-1 2D complex nonuniform FFT.
This computes, to relative precision eps, via a fast algorithm:

                  nj
    f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                 j=1

    for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

 # Inputs
  -  `xj`,`yj`   coordinates of nonuniform sources on the square [-3pi,3pi)^2,
            each a length-nj vector
  -  `cj`      length-nj complex vector of source strengths. If length(cj)>nj,
            expects a stack of vectors (eg, a nj*ntrans matrix) each of which is
            transformed with the same source locations.
  -  `iflag`   if >=0, uses + sign in exponential, otherwise - sign.
   - `eps`     relative precision requested (generally between 1e-15 and 1e-1)
   - `ms`,`mt`   number of Fourier modes requested in x & y; each may be even or odd.
            In either case the mode range is integers lying in [-m/2, (m-1)/2]
  -  kwargs  (optional), see `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 # Output
  - size `(ms,mt)` complex matrix of Fourier coefficients f
            (ordering given by opts.modeord in each dimension; `ms` fast, `mt` slow),
            or, if `ntrans>1`, a array of size `(ms,mt,ntrans)`.
"""
function nufft2d1(xj      :: Array{T},
                  yj      :: Array{T},
                  cj      :: Array{Complex{T}},
                  iflag   :: Integer, 
                  eps     :: Real,
                  ms      :: Integer,
                  mt      :: Integer;
                  kwargs...) where T <: finufftReal
    valid_setpts(1,2,xj,yj)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, ms, mt, ntrans)
    checkkwdtype(T; kwargs...)
    nufft2d1!(xj, yj, cj, iflag, eps, fk;kwargs...)
    return fk
end

"""
    nufft3d1(xj      :: Array{Float64} or Array{Float32}, 
             yj      :: Array{Float64} or Array{Float32}, 
             zj      :: Array{Float64} or Array{Float32}, 
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Real,
             ms      :: Integer,
             mt      :: Integer,
             mu      :: Integer;
             kwargs...
            ) -> Array{ComplexF64}

Compute type-1 3D complex nonuniform FFT.
This computes, to relative precision eps, via a fast algorithm:

                      nj
    f[k1,k2,k3] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
                     j=1

    for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
        -mu/2 <= k3 <= (mu-1)/2.

 # Inputs
 -   `xj`,`yj`,`zj` coordinates of nonuniform sources on the cube [-3pi,3pi)^3,
             each a length-nj vector
 -   `cj`       length-nj complex vector of source strengths. If length(cj)>nj,
             expects a stack of vectors (eg, a nj*ntrans matrix) each of which is
             transformed with the same source locations.
 -   `iflag`    if >=0, uses + sign in exponential, otherwise - sign.
  -  `eps`      relative precision requested (generally between 1e-15 and 1e-1)
 -   `ms`,`mt`,`mu` number of Fourier modes requested in x,y and z; each may be
             even or odd.
             In either case the mode range is integers lying in [-m/2, (m-1)/2]
  -  kwargs  (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 # Output
  - size `(ms,mt,mu)` complex array of Fourier coefficients f
            (ordering given by opts.modeord in each dimension; `ms` fastest, `mu`
            slowest), or, if `ntrans>1`, a 4D array of size `(ms,mt,mu,ntrans)`.
"""
function nufft3d1(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},
                  cj      :: Array{Complex{T}},
                  iflag   :: Integer, 
                  eps     :: Real,
                  ms      :: Integer,
                  mt      :: Integer,
                  mu      :: Integer;
                  kwargs...)  where T <: finufftReal
    valid_setpts(1,3,xj,yj,zj)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, ms, mt, mu, ntrans)
    checkkwdtype(T; kwargs...)
    nufft3d1!(xj, yj, zj, cj, iflag, eps, fk;kwargs...)
    return fk
end


## Type-2

"""
    nufft1d2(xj      :: Array{Float64} or Array{Float32}, 
             iflag   :: Integer, 
             eps     :: Real,
             fk      :: Array{ComplexF64} or Array{ComplexF32};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-2 1D complex nonuniform FFT. 
This computes, to relative precision eps, via a fast algorithm:

    c[j] = SUM   f[k1] exp(+/-i k1 x[j])      for j = 1,...,nj
            k1
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

 # Input
  -  `xj`      location of nonuniform targets on interval [-3pi,3pi), length nj
   - `fk`      complex Fourier coefficients. If a vector, length sets `ms`
            (with mode ordering given by opts.modeord). If a matrix, each
            column is transformed with the same nonuniform targets.
  -  `iflag`   if >=0, uses + sign in exponential, otherwise - sign.
  -  `eps`     relative precision requested (generally between 1e-15 and 1e-1)
  -  kwargs  (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 # Output
  - complex `(nj,)` vector c of answers at targets, or,
           if `ntrans>1`, matrix of size `(nj,ntrans)`.
"""
function nufft1d2(xj      :: Array{T},
                  iflag   :: Integer,
                  eps     :: Real,
                  fk      :: Array{Complex{T}};
                  kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,1,xj)
    (ms, ntrans) = get_nmodes_from_fk(1,fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    checkkwdtype(T; kwargs...)
    nufft1d2!(xj, cj, iflag, eps, fk;kwargs...)
    return cj
end

"""
    nufft2d2(xj      :: Array{Float64} or Array{Float32}, 
             yj      :: Array{Float64} or Array{Float32}, 
             iflag   :: Integer, 
             eps     :: Real,
             fk      :: Array{ComplexF64} or Array{ComplexF32};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-2 2D complex nonuniform FFT. 
This computes, to relative precision eps, via a fast algorithm:

    c[j] =  SUM   f[k1,k2] exp(+/-i (k1 x[j] + k2 y[j]))  for j = 1,..,nj
           k1,k2
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

 # Inputs
   -  `xj`,`yj`   coordinates of nonuniform targets on the square [-3pi,3pi)^2,
            each a vector of length nj
  -  `fk`      complex Fourier coefficient matrix, whose size determines (ms,mt).
            (Mode ordering given by opts.modeord, in each dimension.)
            If a 3D array, 3rd dimension sets `ntrans`, and each of `ntrans`
            matrices is transformed with the same nonuniform targets.
  -  `iflag`   if >=0, uses + sign in exponential, otherwise - sign.
  -  `eps`     relative precision requested (generally between 1e-15 and 1e-1)
 -   kwargs  (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 # Output
   - complex size `(nj,)` vector c of answers at targets, or,
            if `ntrans>1`, matrix of size `(nj,ntrans)`.

"""
function nufft2d2(xj      :: Array{T},
                  yj      :: Array{T},
                  iflag   :: Integer,
                  eps     :: Real,
                  fk      :: Array{Complex{T}};
                  kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,2,xj,yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2,fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    checkkwdtype(T; kwargs...)
    nufft2d2!(xj, yj, cj, iflag, eps, fk;kwargs...)
    return cj
end

"""
    nufft3d2(xj      :: Array{Float64} or Array{Float32}, 
             yj      :: Array{Float64} or Array{Float32}, 
             zj      :: Array{Float64} or Array{Float32}, 
             iflag   :: Integer, 
             eps     :: Real,
             fk      :: Array{ComplexF64} or Array{ComplexF32};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-2 3D complex nonuniform FFT. 
This computes, to relative precision eps, via a fast algorithm:

    c[j] =   SUM   f[k1,k2,k3] exp(+/-i (k1 x[j] + k2 y[j] + k3 z[j]))
           k1,k2,k3
                           for j = 1,..,nj
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,
                      -mu/2 <= k3 <= (mu-1)/2.

 # Inputs
  -  `xj`,`yj`,`zj` coordinates of nonuniform targets on the cube [-3pi,3pi)^3,
             each a vector of length nj
  -  `fk`       complex Fourier coefficient array, whose size sets `(ms,mt,mu)`.
             (Mode ordering given by opts.modeord, in each dimension.)
             If a 4D array, 4th dimension sets `ntrans`, and each of `ntrans`
             3D arrays is transformed with the same nonuniform targets.
  -  `iflag`    if >=0, uses + sign in exponential, otherwise - sign.
 -   `eps`      relative precision requested (generally between 1e-15 and 1e-1)
   - kwargs   (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 #  Output
   - complex vector c of size `(nj,)` giving answers at targets, or,
             if `ntrans>1`, matrix of size `(nj,ntrans)`.
"""
function nufft3d2(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},
                  iflag   :: Integer, 
                  eps     :: Real,
                  fk      :: Array{Complex{T}};
                  kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,3,xj,yj,zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3,fk)
    cj = Array{Complex{T}}(undef, nj, ntrans)
    checkkwdtype(T; kwargs...)
    nufft3d2!(xj, yj, zj, cj, iflag, eps, fk;kwargs...)
    return cj
end


## Type-3

"""
    nufft1d3(xj      :: Array{Float64} or Array{Float32}, 
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Real,
             sk      :: Array{Float64} or Array{Float32};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-3 1D complex nonuniform FFT.
This computes, to relative precision eps, via a fast algorithm:

             nj
    f[k]  =  SUM   c[j] exp(+-i s[k] x[j]),      for k = 1, ..., nk
             j=1

 # Inputs
 -   `xj`       locations of nonuniform sources on R (real line), length-nj vector.
 -   `cj`       length-nj complex vector of source strengths. If length(cj)>nj,
             expects a size `(nj,ntrans)` matrix each column of which is
             transformed with the same source and target locations.
 -    `iflag`    if >=0, uses + sign in exponential, otherwise - sign.
 - `eps`      relative precision requested (generally between 1e-15 and 1e-1)
 -  `sk`       frequency locations of nonuniform targets on R, length-nk vector.
  -  kwargs   (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 #  Output
  - complex vector f size '(nk,)` of values at targets, or, if `ntrans>1`,
             a matrix of size `(nk,ntrans)`
"""
function nufft1d3(xj      :: Array{T},
                  cj      :: Array{Complex{T}},
                  iflag   :: Integer, 
                  eps     :: Real,
                  sk      :: Array{T};
                  kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(3,1,xj,T[],T[],sk)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    checkkwdtype(T; kwargs...)
    nufft1d3!(xj, cj, iflag, eps, sk, fk;kwargs...)
    return fk
end

"""
    nufft2d3(xj      :: Array{Float64} or Array{Float32}, 
             yj      :: Array{Float64} or Array{Float32},
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Real,
             sk      :: Array{Float64} or Array{Float32},
             tk      :: Array{Float64} or Array{Float32};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-3 2D complex nonuniform FFT.
This computes, to relative precision eps, via a fast algorithm:

             nj
    f[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j])),  for k = 1, ..., nk
             j=1

 # Inputs
  -  `xj`,`yj`    coordinates of nonuniform sources in R^2, each a length-nj vector.
   - `cj`    complex vector `(nj,)` of source strengths. If length(cj)>nj,
             expects a `(nj,ntrans)` matrix, each column of which is
             transformed with the same source and target locations.
 -   `iflag`    if >=0, uses + sign in exponential, otherwise - sign.
 -   `eps`      relative precision requested (generally between 1e-15 and 1e-1)
  -  `sk`,`tk`    frequency coordinates of nonuniform targets in R^2,
             each a length-nk vector.
 -   kwargs   (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 # Output
  - complex vector size `(nk,)` of values at targets, or, if `ntrans>1`,
             a matrix of size `(nk,ntrans)`
"""
function nufft2d3(xj      :: Array{T},
                  yj      :: Array{T}, 
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: Real,
                  sk      :: Array{T},
                  tk      :: Array{T};
                  kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(3,2,xj,yj,T[],sk,tk)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    checkkwdtype(T; kwargs...)
    nufft2d3!(xj, yj, cj, iflag, eps, sk, tk, fk;kwargs...)
    return fk
end

"""
    nufft3d3(xj      :: Array{Float64} or Array{Float32}, 
             yj      :: Array{Float64} or Array{Float32},
             zj      :: Array{Float64} or Array{Float32},
             cj      :: Array{ComplexF64} or Array{ComplexF32}, 
             iflag   :: Integer, 
             eps     :: Real,
             sk      :: Array{Float64} or Array{Float32},
             tk      :: Array{Float64} or Array{Float32},
             uk      :: Array{Float64} or Array{Float32};
             kwargs...
            ) -> Array{ComplexF64}

Compute type-3 3D complex nonuniform FFT.
This computes, to relative precision eps, via a fast algorithm:

             nj
    f[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j] + u[k] z[j])),
             j=1
                             for k = 1, ..., nk

 # Inputs
  -  `xj`,`yj`,`zj` coordinates of nonuniform sources in R^3, each a length-nj vector.
 -   `cj`     complex `(nj,)` vector of source strengths. If length(cj)>nj,
             expects a '(nj,ntrans)' matrix, each column of which is
             transformed with the same source and target locations.
 -   `iflag`    if >=0, uses + sign in exponential, otherwise - sign.
 -   `eps`      relative precision requested (generally between 1e-15 and 1e-1)
  -  `sk`,`tk,`uk` frequency coordinates of nonuniform targets in R^3,
             each a length-nk vector.
  -  kwargs   (optional). See `nufft_opts` and https://finufft.readthedocs.io/en/latest/opts.html
 #  Output
  - size `(nk,)` complex vector f values at targets, or, if `ntrans>1`,
             a matrix of size `(nk,ntrans)`
"""
function nufft3d3(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},                   
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: Real,
                  sk      :: Array{T},
                  tk      :: Array{T},
                  uk      :: Array{T};
                  kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(3,3,xj,yj,zj,sk,tk,uk)
    ntrans = valid_ntr(xj,cj)
    fk = Array{Complex{T}}(undef, nk, ntrans)
    checkkwdtype(T; kwargs...)
    nufft3d3!(xj, yj, zj, cj, iflag, eps, sk, tk, uk, fk;kwargs...)
    return fk
end


### Direct interfaces (No allocation)
### Double precision
## 1D

"""
    nufft1d1!(xj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
            )

Compute type-1 1D complex nonuniform FFT. Output written to `fk`. See `nufft1d1`.
"""
function nufft1d1!(xj      :: Array{T},
                   cj      :: Array{Complex{T}},
                   iflag   :: Integer, 
                   eps     :: Real,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    valid_setpts(1,1,xj)
    ntrans = valid_ntr(xj,cj)
    (ms, ntrans_fk) = get_nmodes_from_fk(1,fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,1,[ms;],iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end


"""
    nufft1d2!(xj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
            )

Compute type-2 1D complex nonuniform FFT. Output written to `cj`. See `nufft1d2`.
"""
function nufft1d2!(xj      :: Array{T},
                   cj      :: Array{Complex{T}},
                   iflag   :: Integer, 
                   eps     :: Real,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,1,xj)
    (ms, ntrans) = get_nmodes_from_fk(1,fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,2,[ms;],iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj)
    finufft_exec!(plan,fk,cj)
    ret = finufft_destroy!(plan)
    check_ret(ret)    
end


"""
    nufft1d3!(xj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              sk      :: Array{Float64} or Array{Float32},
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
             )

Compute type-3 1D complex nonuniform FFT. Output written to `fk`. See `nufft1d3`.
"""
function nufft1d3!(xj      :: Array{T},
                   cj      :: Array{Complex{T}},
                   iflag   :: Integer, 
                   eps     :: Real,
                   sk      :: Array{T},
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(3,1,xj,T[],T[],sk)
    ntrans = valid_ntr(xj,cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,3,1,iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,T[],T[],sk)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end


## 2D

"""
    nufft2d1!(xj      :: Array{Float64} or Array{Float32}, 
              yj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
            )

Compute type-1 2D complex nonuniform FFT. Output written to `fk`. See `nufft2d1`.
"""
function nufft2d1!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Real,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    valid_setpts(1,2,xj,yj)
    ntrans = valid_ntr(xj,cj)
    (ms, mt, ntrans_fk) = get_nmodes_from_fk(2,fk)
    @assert ntrans==ntrans_fk

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,1,[ms;mt],iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,yj)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end


"""
    nufft2d2!(xj      :: Array{Float64} or Array{Float32}, 
              yj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
            )

Compute type-2 2D complex nonuniform FFT. Output written to `cj`. See `nufft2d2`.
"""
function nufft2d2!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Real,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(1,2,xj,yj)
    (ms, mt, ntrans) = get_nmodes_from_fk(2,fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,2,[ms;mt],iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,yj)
    finufft_exec!(plan,fk,cj)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end

"""
    nufft2d3!(xj      :: Array{Float64} or Array{Float32}, 
              yj      :: Array{Float64} or Array{Float32},
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              sk      :: Array{Float64} or Array{Float32},
              tk      :: Array{Float64} or Array{Float32},
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
             )

Compute type-3 2D complex nonuniform FFT. Output written to 'fk'. See `nufft2d3`.
"""
function nufft2d3!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Real,
                   sk      :: Array{T},
                   tk      :: Array{T},
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(3,2,xj,yj,T[],sk,tk)
    ntrans = valid_ntr(xj,cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,3,2,iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,yj,T[],sk,tk)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end

## 3D

"""
    nufft3d1!(xj      :: Array{Float64} or Array{Float32}, 
              yj      :: Array{Float64} or Array{Float32}, 
              zj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
            )

Compute type-1 3D complex nonuniform FFT. Output written to `fk`. See `nufft3d1`.
"""
function nufft3d1!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   zj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Real,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    valid_setpts(1,3,xj,yj,zj)
    ntrans = valid_ntr(xj,cj)
    (ms, mt, mu, ntrans_fk) = get_nmodes_from_fk(3,fk)
    @assert ntrans == ntrans_fk

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,1,[ms;mt;mu],iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,yj,zj)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end

"""
    nufft3d2!(xj      :: Array{Float64} or Array{Float32}, 
              yj      :: Array{Float64} or Array{Float32}, 
              zj      :: Array{Float64} or Array{Float32}, 
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
            )

Compute type-2 3D complex nonuniform FFT. Output written to `cj`. See `nufft3d2`.
"""
function nufft3d2!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   zj      :: Array{T},                    
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Real,
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(2,3,xj,yj,zj)
    (ms, mt, mu, ntrans) = get_nmodes_from_fk(3,fk)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,2,[ms;mt;mu],iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,yj,zj)
    finufft_exec!(plan,fk,cj)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end

"""
    nufft3d3!(xj      :: Array{Float64} or Array{Float32}, 
              yj      :: Array{Float64} or Array{Float32},
              zj      :: Array{Float64} or Array{Float32},
              cj      :: Array{ComplexF64} or Array{ComplexF32}, 
              iflag   :: Integer, 
              eps     :: Real,
              sk      :: Array{Float64} or Array{Float32},
              tk      :: Array{Float64} or Array{Float32},
              uk      :: Array{Float64} or Array{Float32},
              fk      :: Array{ComplexF64} or Array{ComplexF32};
              kwargs...
             )

Compute type-3 3D complex nonuniform FFT. Output written to `fk`. See `nufft3d3`.
"""
function nufft3d3!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   zj      :: Array{T},                   
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Real,
                   sk      :: Array{T},
                   tk      :: Array{T},
                   uk      :: Array{T},
                   fk      :: Array{Complex{T}};
                   kwargs...) where T <: finufftReal
    (nj, nk) = valid_setpts(3,3,xj,yj,zj,sk,tk,uk)
    ntrans = valid_ntr(xj,cj)

    checkkwdtype(T; kwargs...)
    plan = finufft_makeplan(T,3,3,iflag,ntrans,eps;kwargs...)
    finufft_setpts!(plan,xj,yj,zj,sk,tk,uk)
    finufft_exec!(plan,cj,fk)
    ret = finufft_destroy!(plan)
    check_ret(ret)
end
