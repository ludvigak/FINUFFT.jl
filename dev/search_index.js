var documenterSearchIndex = {"docs":
[{"location":"#FINUFFT.jl-Reference","page":"FINUFFT.jl Reference","title":"FINUFFT.jl Reference","text":"","category":"section"},{"location":"","page":"FINUFFT.jl Reference","title":"FINUFFT.jl Reference","text":"For installation and basic usage, see the README at https://github.com/ludvigak/FINUFFT.jl","category":"page"},{"location":"","page":"FINUFFT.jl Reference","title":"FINUFFT.jl Reference","text":"For documentation of the library functions that are being called, see the FINUFFT documentation at https://finufft.readthedocs.io","category":"page"},{"location":"#Index","page":"FINUFFT.jl Reference","title":"Index","text":"","category":"section"},{"location":"","page":"FINUFFT.jl Reference","title":"FINUFFT.jl Reference","text":"","category":"page"},{"location":"#Types","page":"FINUFFT.jl Reference","title":"Types","text":"","category":"section"},{"location":"","page":"FINUFFT.jl Reference","title":"FINUFFT.jl Reference","text":"Modules = [FINUFFT]\nOrder = [:type]","category":"page"},{"location":"#FINUFFT.nufft_opts","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft_opts","text":"mutable struct nufft_opts    \n    modeord            :: Cint\n    chkbnds            :: Cint\n    debug              :: Cint\n    spread_debug       :: Cint\n    showwarn           :: Cint\n    nthreads           :: Cint\n    fftw               :: Cint\n    spread_sort        :: Cint\n    spread_kerevalmeth :: Cint\n    spread_kerpad      :: Cint\n    upsampfac          :: Cdouble\n    spread_thread      :: Cint\n    maxbatchsize       :: Cint\n    spread_nthr_atomic :: Cint\n    spread_max_sp_size :: Cint\nend\n\nOptions struct passed to the FINUFFT library.\n\nFields\n\nThis is a summary only; see FINUFFT documentation for full descriptions.\n\nmodeord :: Cint\n\n0: CMCL-style increasing mode ordering (neg to pos), or\n1: FFT-style mode ordering (affects type-1,2 only)\n\nchkbnds :: Cint\n\n0: don't check if input NU pts in [-3pi,3pi], 1: do\n\ndebug :: Cint\n\n0: silent, 1: text basic timing output\n\nspread_debug :: Cint\n\npassed to spread_opts, 0 (no text) 1 (some) or 2 (lots)\n\nshowwarn :: Cint\n\nWhether to print warnings to stderr. 0: silent, 1: print warnings\n\nnthreads :: Cint\n\nHow many threads FINUFFT should use, or 0 (use max available in OMP)\n\nfftw :: Cint\n\n0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster FFTs)\n\nspread_sort :: Cint\n\npassed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)\n\nspread_kerevalmeth :: Cint\n\npassed to spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)\n\nspread_kerpad :: Cint\n\npassed to spread_opts, 0: don't pad to mult of 4, 1: do\n\nupsampfac :: Cdouble\n\nupsampling ratio sigma: 2.0 (standard), or 1.25 (small FFT), or\n0.0 (auto).\n\nspread_thread :: Cint\n\n(for ntrans>1 only)\n0: auto choice,\n1: sequential multithreaded,\n2: parallel singlethreaded spread.\n\nmaxbatchsize :: Cint\n\n(for ntrans>1 only). max blocking size for vectorized, 0 for auto-set\n\nspread_nthr_atomic :: Cint\n\nif >=0, threads above which spreader OMP critical goes atomic\n\nspread_max_sp_size :: Cint\n\nif >0, overrides spreader (dir=1 only) max subproblem size\n\n\n\n\n\n","category":"type"},{"location":"#Functions","page":"FINUFFT.jl Reference","title":"Functions","text":"","category":"section"},{"location":"","page":"FINUFFT.jl Reference","title":"FINUFFT.jl Reference","text":"Modules = [FINUFFT]\nOrder = [:function]","category":"page"},{"location":"#FINUFFT.finufft_default_opts","page":"FINUFFT.jl Reference","title":"FINUFFT.finufft_default_opts","text":"p = finufft_default_opts()\np = finufft_default_opts(dtype=Float32)\n\nReturn a nufft_opts struct with the default FINUFFT settings. Set up the double precision variant by default.\nSee: https://finufft.readthedocs.io/en/latest/usage.html#options\n\n\n\n\n\n","category":"function"},{"location":"#FINUFFT.finufft_destroy-Union{Tuple{finufft_plan{T}}, Tuple{T}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.finufft_destroy","text":"status = finufft_destroy(plan::finufft_plan{T}) where T <: finufftReal\n\nThis destroys a FINUFFT plan object: it deallocates all stored FFTW plans, nonuniform point sorting arrays, kernel Fourier transforms arrays, and any other allocations, and nulls the plan pointer.\n\nAn integer status code is returned, that is 0 if successful. If one attempts to destroy an already-destroyed plan, 1 is returned (see FINUFFT documentation for finufft_destroy).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.finufft_exec!-Union{Tuple{T}, Tuple{finufft_plan{T}, Array{Complex{T}, N} where N, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.finufft_exec!","text":"finufft_exec!(plan::finufft_plan{T},\n                  input::Array{Complex{T}},\n                  output::Array{Complex{T}}) where T <: finufftReal\n\nExecute single or many-vector FINUFFT transforms in a plan, with output written to preallocated array. See finufft_exec for arguments.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.finufft_exec-Union{Tuple{T}, Tuple{finufft_plan{T}, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.finufft_exec","text":"output::Array{Complex{T}} = finufft_exec(plan::finufft_plan{T},\n                  input::Array{Complex{T}}) where T <: finufftReal\n\nExecute single or many-vector FINUFFT transforms in a plan.\n\noutput = finufft_exec(plan, input)\n\nFor plan a previously created finufft_plan object also containing all   needed nonuniform point coordinates, do a single (or if ntrans>1 in the   plan stage, multiple) NUFFT transform(s), with the strengths or Fourier   coefficient inputs vector(s) from input. The result of the transform(s)   is returned as a (possibly multidimensional) array.\n\nInputs\n\n- `plan`     `finufft_plan` object, already planned and containing\nnonuniform points.\n- `input`    strengths (types 1 or 3) or Fourier coefficients (type 2)\n          vector, matrix, or array of appropriate size. For type 1 and 3,\n          this is either a length-M vector (where M is the length of `xj`),\n          or an `(M,ntrans)` matrix when `ntrans>1`. For type 2, in 1D this is size `(ms,)`, in 2D size `(ms,mt)`, or in 3D size `(ms,mt,mu)`, or\n          each of these with an extra last dimension `ntrans` if `ntrans>1`.\n\nOutput\n\n `output`   vector of output strengths at targets (types 2 or 3), or array\n          of Fourier coefficients (type 1), or, if `ntrans>1`, a stack of\n          such vectors or arrays, of appropriate size.\n          Specifically, if `ntrans=1`, for type 1, in 1D\n          this is size `(ms,)`, in 2D size\n          `(ms,mt)`, or in 3D size `(ms,mt,mu)`; for types 2 and 3\n          it is a column vector of length `M` (the length of `xj` in type 2),\n          or `nk` (the length of `s` in type 3). If `ntrans>1` it is a stack\n          of such objects, ie, it has an extra last dimension `ntrans`.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.finufft_makeplan-Tuple{Integer, Union{Integer, Array{Int64, N} where N}, Integer, Integer, Real}","page":"FINUFFT.jl Reference","title":"FINUFFT.finufft_makeplan","text":"finufft_makeplan(type::Integer,\n                      n_modes_or_dim::Union{Array{Int64},Integer},\n                      iflag::Integer,\n                      ntrans::Integer,\n                      eps::Real;\n                      dtype=Float64,\n                      kwargs...)\n\nCreates a finufft_plan object for the guru interface to FINUFFT, of  type 1, 2 or 3, and with given numbers of Fourier modes (unless type 3).\n\nInputs\n\ntype            transform type: 1, 2, or 3\nn_modes_or_dim  if type is 1 or 2, the number of Fourier modes in each                 dimension: ms in 1D, [ms mt] in 2D, or [ms mt mu] in 3D.                 Its length thus sets the dimension, which must be 1, 2 or 3.                 If type is 3, in contrast, its value fixes the dimension.\niflag   if >=0, uses + sign in exponential, otherwise - sign.\nntrans          number of transforms to compute simultaneously\neps     relative precision requested (generally between 1e-15 and 1e-1),         real, need not match type of dtype\ndtype           Float32 or Float64, plan for single precision or double precision\nkwargs  (optional): for options, see nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nReturns\n\nfinufft_plan struct\n\nExamples\n\njulia> p = finufft_makeplan(2,10,+1,1,1e-6);\n\ncreates a plan for a 1D type 2 Float64 transform with 10 Fourier modes and tolerance 1e-6.\n\njulia> p = finufft_makeplan(1,[10 20],+1,1,1e-6);\n\ncreates a plan for a 2D type 1 Float64 transform with 10*20 Fourier modes.\n\njulia> p = finufft_makeplan(3,2,+1,1,1e-4,dtype=Float32,nthreads=4);\n\ncreates a plan for a 2D type 3 Float32 transform with tolerance 1e-4, to use 4 threads.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.finufft_setpts-Union{Tuple{T}, Tuple{finufft_plan{T}, Array{T, N} where N}, Tuple{finufft_plan{T}, Array{T, N} where N, Array{T, N} where N}, Tuple{finufft_plan{T}, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N}, Tuple{finufft_plan{T}, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N}, Tuple{finufft_plan{T}, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N}, Tuple{finufft_plan{T}, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.finufft_setpts","text":"finufft_setpts(plan, xj [, yj[, zj[, s[, t[, u]]]]])\n\nInput nonuniform points for general FINUFFT transform(s).\n\nGiven an already-planned finufft_plan, this reads in nonuniform point  coordinate arrays xj (and yj if 2D or 3D, and zj if 3D), and additionally in the type 3 case, nonuniform  frequency target coordinate arrays s (and t if 2D or 3D, and u if 3D). Empty arrays may be passed in the case of  unused dimensions. For all types, sorting is done to internally store a  reindexing of points, and for type 3 the spreading and FFTs are planned.  These nonuniform points may then be used for multiple transforms.\n\nInputs\n\nplan   a finufft_plan object for one/many general nonuniform FFTs\nxj Array{Float32} or Array{Float64}, vector of x-coords of all nonuniform points\nyj     empty (if dim<2), or vector of y-coords of all nonuniform points\nzj     empty (if dim<3), or vector of z-coords of all nonuniform points\ns      vector of x-coords of all nonuniform frequency targets\nt      empty (if dim<2), or vector of y-coords of all frequency targets\nu      empty (if dim<3), or vector of z-coords of all frequency targets\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft1d1!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft1d1!","text":"nufft1d1!(xj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n        )\n\nCompute type-1 1D complex nonuniform FFT. Output written to fk. See nufft1d1.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft1d1-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Integer}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft1d1","text":"nufft1d1(xj      :: Array{Float64} or Array{Float32}, \n         cj      :: Array{ComplexF64} or Array{ComplexF32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         ms      :: Integer;\n         kwargs...\n        ) -> Array{ComplexF64} or Array{ComplexF32}\n\nCompute type-1 1D complex nonuniform FFT.  This computes, to relative precision eps, via a fast algorithm:\n\n          nj\nf(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2\n         j=1\n\nInputs\n\nxj      locations of nonuniform sources on interval [-3pi,3pi), length nj\ncj      length-nj complex vector of source strengths. If length(cj)>nj,         expects a stack of vectors (eg, a nj*ntrans matrix) each of which is         transformed with the same source locations.\niflag   if >=0, uses + sign in exponential, otherwise - sign.\neps     relative precision requested (generally between 1e-15 and 1e-1)\nms      number of Fourier modes computed, may be even or odd;         in either case, mode range is integers lying in [-ms/2, (ms-1)/2]\nkwargs  (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\nsize (ms,) complex vector of Fourier coefficients f, or, if         ntrans>1, matrix of size (ms,ntrans).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft1d2!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft1d2!","text":"nufft1d2!(xj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n        )\n\nCompute type-2 1D complex nonuniform FFT. Output written to cj. See nufft1d2.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft1d2-Union{Tuple{T}, Tuple{Array{T, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft1d2","text":"nufft1d2(xj      :: Array{Float64} or Array{Float32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         fk      :: Array{ComplexF64} or Array{ComplexF32};\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-2 1D complex nonuniform FFT.  This computes, to relative precision eps, via a fast algorithm:\n\nc[j] = SUM   f[k1] exp(+/-i k1 x[j])      for j = 1,...,nj\n        k1\n where sum is over -ms/2 <= k1 <= (ms-1)/2.\n\nInput\n\nxj      location of nonuniform targets on interval [-3pi,3pi), length nj\nfk      complex Fourier coefficients. If a vector, length sets ms         (with mode ordering given by opts.modeord). If a matrix, each         column is transformed with the same nonuniform targets.\niflag   if >=0, uses + sign in exponential, otherwise - sign.\neps     relative precision requested (generally between 1e-15 and 1e-1)\nkwargs  (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\ncomplex (nj,) vector c of answers at targets, or,        if ntrans>1, matrix of size (nj,ntrans).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft1d3!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{T, N} where N, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft1d3!","text":"nufft1d3!(xj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          sk      :: Array{Float64} or Array{Float32},\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n         )\n\nCompute type-3 1D complex nonuniform FFT. Output written to fk. See nufft1d3.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft1d3-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{T, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft1d3","text":"nufft1d3(xj      :: Array{Float64} or Array{Float32}, \n         cj      :: Array{ComplexF64} or Array{ComplexF32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         sk      :: Array{Float64} or Array{Float32};\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-3 1D complex nonuniform FFT. This computes, to relative precision eps, via a fast algorithm:\n\n         nj\nf[k]  =  SUM   c[j] exp(+-i s[k] x[j]),      for k = 1, ..., nk\n         j=1\n\nInputs\n\nxj       locations of nonuniform sources on R (real line), length-nj vector.\ncj       length-nj complex vector of source strengths. If length(cj)>nj,           expects a size (nj,ntrans) matrix each column of which is           transformed with the same source and target locations.\niflag    if >=0, uses + sign in exponential, otherwise - sign.\neps      relative precision requested (generally between 1e-15 and 1e-1)\nsk       frequency locations of nonuniform targets on R, length-nk vector.\nkwargs   (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\ncomplex vector f size '(nk,)of values at targets, or, ifntrans>1,          a matrix of size(nk,ntrans)`\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft2d1!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft2d1!","text":"nufft2d1!(xj      :: Array{Float64} or Array{Float32}, \n          yj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n        )\n\nCompute type-1 2D complex nonuniform FFT. Output written to fk. See nufft2d1.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft2d1-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Integer, Integer}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft2d1","text":"nufft2d1(xj      :: Array{Float64} or Array{Float32}\n         yj      :: Array{Float64} or Array{Float32}, \n         cj      :: Array{ComplexF64} or Array{ComplexF32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         ms      :: Integer,\n         mt      :: Integer;\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-1 2D complex nonuniform FFT. This computes, to relative precision eps, via a fast algorithm:\n\n              nj\nf[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))\n             j=1\n\nfor -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.\n\nInputs\n\nxj,yj   coordinates of nonuniform sources on the square [-3pi,3pi)^2,         each a length-nj vector\ncj      length-nj complex vector of source strengths. If length(cj)>nj,         expects a stack of vectors (eg, a nj*ntrans matrix) each of which is         transformed with the same source locations.\niflag   if >=0, uses + sign in exponential, otherwise - sign.\neps     relative precision requested (generally between 1e-15 and 1e-1)\nms,mt   number of Fourier modes requested in x & y; each may be even or odd.         In either case the mode range is integers lying in [-m/2, (m-1)/2]\nkwargs  (optional), see nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\nsize (ms,mt) complex matrix of Fourier coefficients f         (ordering given by opts.modeord in each dimension; ms fast, mt slow),         or, if ntrans>1, a array of size (ms,mt,ntrans).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft2d2!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft2d2!","text":"nufft2d2!(xj      :: Array{Float64} or Array{Float32}, \n          yj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n        )\n\nCompute type-2 2D complex nonuniform FFT. Output written to cj. See nufft2d2.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft2d2-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft2d2","text":"nufft2d2(xj      :: Array{Float64} or Array{Float32}, \n         yj      :: Array{Float64} or Array{Float32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         fk      :: Array{ComplexF64} or Array{ComplexF32};\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-2 2D complex nonuniform FFT.  This computes, to relative precision eps, via a fast algorithm:\n\nc[j] =  SUM   f[k1,k2] exp(+/-i (k1 x[j] + k2 y[j]))  for j = 1,..,nj\n       k1,k2\n where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,\n\nInputs\n\nxj,yj   coordinates of nonuniform targets on the square [-3pi,3pi)^2,        each a vector of length nj\nfk      complex Fourier coefficient matrix, whose size determines (ms,mt).        (Mode ordering given by opts.modeord, in each dimension.)        If a 3D array, 3rd dimension sets ntrans, and each of ntrans        matrices is transformed with the same nonuniform targets.\niflag   if >=0, uses + sign in exponential, otherwise - sign.\neps     relative precision requested (generally between 1e-15 and 1e-1)\nkwargs  (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\ncomplex size (nj,) vector c of answers at targets, or,        if ntrans>1, matrix of size (nj,ntrans).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft2d3!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft2d3!","text":"nufft2d3!(xj      :: Array{Float64} or Array{Float32}, \n          yj      :: Array{Float64} or Array{Float32},\n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          sk      :: Array{Float64} or Array{Float32},\n          tk      :: Array{Float64} or Array{Float32},\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n         )\n\nCompute type-3 2D complex nonuniform FFT. Output written to 'fk'. See nufft2d3.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft2d3-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{T, N} where N, Array{T, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft2d3","text":"nufft2d3(xj      :: Array{Float64} or Array{Float32}, \n         yj      :: Array{Float64} or Array{Float32},\n         cj      :: Array{ComplexF64} or Array{ComplexF32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         sk      :: Array{Float64} or Array{Float32},\n         tk      :: Array{Float64} or Array{Float32};\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-3 2D complex nonuniform FFT. This computes, to relative precision eps, via a fast algorithm:\n\n         nj\nf[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j])),  for k = 1, ..., nk\n         j=1\n\nInputs\n\nxj,yj    coordinates of nonuniform sources in R^2, each a length-nj vector.\ncj    complex vector (nj,) of source strengths. If length(cj)>nj,          expects a (nj,ntrans) matrix, each column of which is          transformed with the same source and target locations.\niflag    if >=0, uses + sign in exponential, otherwise - sign.\neps      relative precision requested (generally between 1e-15 and 1e-1)\nsk,tk    frequency coordinates of nonuniform targets in R^2,          each a length-nk vector.\nkwargs   (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\ncomplex vector size (nk,) of values at targets, or, if ntrans>1,          a matrix of size (nk,ntrans)\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft3d1!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft3d1!","text":"nufft3d1!(xj      :: Array{Float64} or Array{Float32}, \n          yj      :: Array{Float64} or Array{Float32}, \n          zj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n        )\n\nCompute type-1 3D complex nonuniform FFT. Output written to fk. See nufft3d1.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft3d1-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Integer, Integer, Integer}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft3d1","text":"nufft3d1(xj      :: Array{Float64} or Array{Float32}, \n         yj      :: Array{Float64} or Array{Float32}, \n         zj      :: Array{Float64} or Array{Float32}, \n         cj      :: Array{ComplexF64} or Array{ComplexF32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         ms      :: Integer,\n         mt      :: Integer,\n         mu      :: Integer;\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-1 3D complex nonuniform FFT. This computes, to relative precision eps, via a fast algorithm:\n\n                  nj\nf[k1,k2,k3] =    SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))\n                 j=1\n\nfor -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,\n    -mu/2 <= k3 <= (mu-1)/2.\n\nInputs\n\nxj,yj,zj coordinates of nonuniform sources on the cube [-3pi,3pi)^3,           each a length-nj vector\ncj       length-nj complex vector of source strengths. If length(cj)>nj,           expects a stack of vectors (eg, a nj*ntrans matrix) each of which is           transformed with the same source locations.\niflag    if >=0, uses + sign in exponential, otherwise - sign.\neps      relative precision requested (generally between 1e-15 and 1e-1)\nms,mt,mu number of Fourier modes requested in x,y and z; each may be           even or odd.           In either case the mode range is integers lying in [-m/2, (m-1)/2]\nkwargs  (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\nsize (ms,mt,mu) complex array of Fourier coefficients f         (ordering given by opts.modeord in each dimension; ms fastest, mu         slowest), or, if ntrans>1, a 4D array of size (ms,mt,mu,ntrans).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft3d2!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft3d2!","text":"nufft3d2!(xj      :: Array{Float64} or Array{Float32}, \n          yj      :: Array{Float64} or Array{Float32}, \n          zj      :: Array{Float64} or Array{Float32}, \n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n        )\n\nCompute type-2 3D complex nonuniform FFT. Output written to cj. See nufft3d2.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft3d2-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Integer, Real, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft3d2","text":"nufft3d2(xj      :: Array{Float64} or Array{Float32}, \n         yj      :: Array{Float64} or Array{Float32}, \n         zj      :: Array{Float64} or Array{Float32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         fk      :: Array{ComplexF64} or Array{ComplexF32};\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-2 3D complex nonuniform FFT.  This computes, to relative precision eps, via a fast algorithm:\n\nc[j] =   SUM   f[k1,k2,k3] exp(+/-i (k1 x[j] + k2 y[j] + k3 z[j]))\n       k1,k2,k3\n                       for j = 1,..,nj\n where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,\n                  -mu/2 <= k3 <= (mu-1)/2.\n\nInputs\n\nxj,yj,zj coordinates of nonuniform targets on the cube [-3pi,3pi)^3,          each a vector of length nj\nfk       complex Fourier coefficient array, whose size sets (ms,mt,mu).          (Mode ordering given by opts.modeord, in each dimension.)          If a 4D array, 4th dimension sets ntrans, and each of ntrans          3D arrays is transformed with the same nonuniform targets.\niflag    if >=0, uses + sign in exponential, otherwise - sign.\neps      relative precision requested (generally between 1e-15 and 1e-1)\nkwargs   (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\ncomplex vector c of size (nj,) giving answers at targets, or,         if ntrans>1, matrix of size (nj,ntrans).\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft3d3!-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft3d3!","text":"nufft3d3!(xj      :: Array{Float64} or Array{Float32}, \n          yj      :: Array{Float64} or Array{Float32},\n          zj      :: Array{Float64} or Array{Float32},\n          cj      :: Array{ComplexF64} or Array{ComplexF32}, \n          iflag   :: Integer, \n          eps     :: Real,\n          sk      :: Array{Float64} or Array{Float32},\n          tk      :: Array{Float64} or Array{Float32},\n          uk      :: Array{Float64} or Array{Float32},\n          fk      :: Array{ComplexF64} or Array{ComplexF32};\n          kwargs...\n         )\n\nCompute type-3 3D complex nonuniform FFT. Output written to fk. See nufft3d3.\n\n\n\n\n\n","category":"method"},{"location":"#FINUFFT.nufft3d3-Union{Tuple{T}, Tuple{Array{T, N} where N, Array{T, N} where N, Array{T, N} where N, Array{Complex{T}, N} where N, Integer, Real, Array{T, N} where N, Array{T, N} where N, Array{T, N} where N}} where T<:Union{Float32, Float64}","page":"FINUFFT.jl Reference","title":"FINUFFT.nufft3d3","text":"nufft3d3(xj      :: Array{Float64} or Array{Float32}, \n         yj      :: Array{Float64} or Array{Float32},\n         zj      :: Array{Float64} or Array{Float32},\n         cj      :: Array{ComplexF64} or Array{ComplexF32}, \n         iflag   :: Integer, \n         eps     :: Real,\n         sk      :: Array{Float64} or Array{Float32},\n         tk      :: Array{Float64} or Array{Float32},\n         uk      :: Array{Float64} or Array{Float32};\n         kwargs...\n        ) -> Array{ComplexF64}\n\nCompute type-3 3D complex nonuniform FFT. This computes, to relative precision eps, via a fast algorithm:\n\n         nj\nf[k]  =  SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j] + u[k] z[j])),\n         j=1\n                         for k = 1, ..., nk\n\nInputs\n\nxj,yj,zj coordinates of nonuniform sources in R^3, each a length-nj vector.\ncj     complex (nj,) vector of source strengths. If length(cj)>nj,          expects a '(nj,ntrans)' matrix, each column of which is          transformed with the same source and target locations.\niflag    if >=0, uses + sign in exponential, otherwise - sign.\neps      relative precision requested (generally between 1e-15 and 1e-1)\nsk,tk,uk` frequency coordinates of nonuniform targets in R^3,          each a length-nk vector.\nkwargs   (optional). See nufft_opts and https://finufft.readthedocs.io/en/latest/opts.html\n\nOutput\n\nsize (nk,) complex vector f values at targets, or, if ntrans>1,          a matrix of size (nk,ntrans)\n\n\n\n\n\n","category":"method"}]
}
