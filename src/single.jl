### Single precision interfaces (only direct calls implemented)


## 1D

"""
    nufftf1d1!(xj      :: Array{Float32},
               cj      :: Array{ComplexF32},
               iflag   :: Integer,
               eps     :: Float32,
               fk      :: Array{ComplexF32}
               [, opts :: nufft_opts]
             )

Single precision type-1 1D complex nonuniform FFT. Output stored in fk.
"""
function nufftf1d1!(xj      :: Array{Float32},
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
    nufftf1d2!(xj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               fk      :: Array{ComplexF32} 
               [, opts :: nufft_opts]
             )

Single precision type-2 1D complex nonuniform FFT. Output stored in cj.
"""
function nufftf1d2!(xj      :: Array{Float32}, 
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
    nufftf1d3!(xj      :: Array{Float32}, 
               cj      :: Array{ComplexF32}, 
               iflag   :: Integer, 
               eps     :: Float32,
               sk      :: Array{Float32},
               fk      :: Array{ComplexF32},
               [, opts :: nufft_opts]
              )

Single precision type-3 1D complex nonuniform FFT. Output stored in fk.
"""
function nufftf1d3!(xj      :: Array{Float32}, 
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
