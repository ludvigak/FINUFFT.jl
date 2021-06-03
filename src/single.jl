### Single precision interfaces (only direct calls implemented)


## 1D

"""
    nufft1d1!(xj      :: Array{Float32},
               cj      :: Array{ComplexF32},
               iflag   :: Integer,
               eps     :: Float32,
               fk      :: Array{ComplexF32}
               [, opts :: nufft_opts]
             )

Single precision type-1 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d1!(xj      :: Array{Float32},
                    cj      :: Array{ComplexF32},
                    iflag   :: Integer,
                    eps     :: Float32,
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj
    ms = length(fk)
    ret = ccall( (:finufftf1d1, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, cj, iflag, eps, ms, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft1d2!(xj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               fk      :: Array{ComplexF32} 
               [, opts :: nufft_opts]
             )

Single precision type-2 1D complex nonuniform FFT. Output stored in cj.
"""
function nufft1d2!(xj      :: Array{Float32}, 
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    ms = length(fk)    
    ret = ccall( (:finufftf1d2, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, cj, iflag, eps, ms, fk, opts
                 )
    check_ret(ret)    
end


"""
    nufft1d3!(xj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               sk      :: Array{Float32},
               fk      :: Array{ComplexF32},
               [, opts :: nufft_opts]
              )

Single precision type-3 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d3!(xj      :: Array{Float32}, 
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    sk      :: Array{Float32},
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    @assert length(fk)==nk
    ret = ccall( (:finufftf1d3, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  Ref{Float32},            
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, cj, iflag, eps, nk, sk, fk, opts
                 )
    check_ret(ret)
end

## 2D

"""
    nufft2d1!(xj      :: Array{Float32}, 
               yj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               fk      :: Array{ComplexF32} 
               [, opts :: nufft_opts]
             )

Single precision type-1 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d1!(xj      :: Array{Float32}, 
                    yj      :: Array{Float32}, 
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    ret = ccall( (:finufftf2d1, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{Float32},            
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  BIGINT,            
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                 )
    check_ret(ret)
end


"""
    nufft2d2!(xj      :: Array{Float32}, 
               yj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               fk      :: Array{ComplexF32} 
               [, opts :: nufft_opts]
             )

Single precision type-2 2D complex nonuniform FFT. Output stored in cj.
"""
function nufft2d2!(xj      :: Array{Float32}, 
                    yj      :: Array{Float32}, 
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    ret = ccall( (:finufftf2d2, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{Float32},            
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  BIGINT,            
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft2d3!(xj      :: Array{Float32}, 
               yj      :: Array{Float32},
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               sk      :: Array{Float32},
               tk      :: Array{Float32},
               fk      :: Array{ComplexF32}
               [, opts :: nufft_opts]
              )

Single precision type-3 2D complex nonuniform FFT. Output stored in fk.
    """
function nufft2d3!(xj      :: Array{Float32}, 
                    yj      :: Array{Float32},
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    sk      :: Array{Float32},
                    tk      :: Array{Float32},
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(fk)==nk    
    ret = ccall( (:finufftf2d3, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{Float32},            
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  Ref{Float32},
                  Ref{Float32},            
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, yj, cj, iflag, eps, nk, sk, tk, fk, opts
                 )
    check_ret(ret)
end

## 3D

"""
    nufft3d1!(xj      :: Array{Float32}, 
               yj      :: Array{Float32}, 
               zj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               fk      :: Array{ComplexF32} 
               [, opts :: nufft_opts]
             )

Single precision type-1 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d1!(xj      :: Array{Float32}, 
                    yj      :: Array{Float32}, 
                    zj      :: Array{Float32}, 
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    ret = ccall( (:finufftf3d1, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{Float32},
                  Ref{Float32},                
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  BIGINT,
                  BIGINT,
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft3d2!(xj      :: Array{Float32}, 
               yj      :: Array{Float32}, 
               zj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               fk      :: Array{ComplexF32} 
               [, opts :: nufft_opts]
             )

Single precision type-2 3D complex nonuniform FFT. Output stored in cj.
"""
function nufft3d2!(xj      :: Array{Float32}, 
                    yj      :: Array{Float32},
                    zj      :: Array{Float32},                    
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    ret = ccall( (:finufftf3d2, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{Float32},            
                  Ref{Float32},            
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  BIGINT,
                  BIGINT,
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft3d3!(xj      :: Array{Float32}, 
               yj      :: Array{Float32},
               zj      :: Array{Float32},
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               sk      :: Array{Float32},
               tk      :: Array{Float32},
               uk      :: Array{Float32},
               fk      :: Array{ComplexF32}
               [, opts :: nufft_opts]
              )

Single precision type-3 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d3!(xj      :: Array{Float32}, 
                    yj      :: Array{Float32},
                    zj      :: Array{Float32},                   
                    cj      :: Array{ComplexF32}, 
                    iflag   :: Integer, 
                    eps     :: Float32,
                    sk      :: Array{Float32},
                    tk      :: Array{Float32},
                    uk      :: Array{Float32},
                    fk      :: Array{ComplexF32},
                    opts    :: nufft_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(uk)==nk    
    @assert length(fk)==nk    
    ret = ccall( (:finufftf3d3, libfinufft),
                 Cint,
                 (BIGINT,
                  Ref{Float32},
                  Ref{Float32},
                  Ref{Float32},                  
                  Ref{ComplexF32},
                  Cint,
                  Float32,
                  BIGINT,
                  Ref{Float32},
                  Ref{Float32},
                  Ref{Float32},                        
                  Ref{ComplexF32},
                  Ref{nufft_opts}),
                 nj, xj, yj, zj, cj, iflag, eps, nk, sk, tk, uk, fk, opts
                 )
    check_ret(ret)
end


# keep nufftf calls for compability
const nufftf1d1 = nufft1d1
const nufftf1d2 = nufft1d2
const nufftf1d3 = nufft1d3
const nufftf2d1 = nufft2d1
const nufftf2d2 = nufft2d2
const nufftf2d3 = nufft2d3
const nufftf3d1 = nufft3d1
const nufftf3d2 = nufft3d2
const nufftf3d3 = nufft3d3

const nufftf1d1! = nufft1d1!
const nufftf1d2! = nufft1d2!
const nufftf1d3! = nufft1d3!
const nufftf2d1! = nufft2d1!
const nufftf2d2! = nufft2d2!
const nufftf2d3! = nufft2d3!
const nufftf3d1! = nufft3d1!
const nufftf3d2! = nufft3d2!
const nufftf3d3! = nufft3d3!
