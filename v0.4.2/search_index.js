var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.jl Reference",
    "category": "page",
    "text": ""
},

{
    "location": "#FINUFFT.jl-Reference-1",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.jl Reference",
    "category": "section",
    "text": "For installation and basic usage, see the README at https://github.com/ludvigak/FINUFFT.jlFor documentation of the library functions that are being called, see the FINUFFT documentation at https://finufft.readthedocs.io/en/latest/usage.html"
},

{
    "location": "#Index-1",
    "page": "FINUFFT.jl Reference",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "#FINUFFT.nufft_opts",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft_opts",
    "category": "type",
    "text": "mutable struct nufft_opts    \n    debug              :: Cint                \n    spread_debug       :: Cint         \n    spread_sort        :: Cint          \n    spread_kerevalmeth :: Cint   \n    spread_kerpad      :: Cint        \n    chkbnds            :: Cint              \n    fftw               :: Cint                 \n    modeord            :: Cint\n    upsampfac          :: Cdouble         \nend\n\nOptions struct passed to the FINUFFT library.\n\nFields\n\ndebug :: Cint\n\n0: silent, 1: text basic timing output\n\nspread_debug :: Cint\n\npassed to spread_opts, 0 (no text) 1 (some) or 2 (lots)\n\nspread_sort :: Cint\n\npassed to spread_opts, 0 (don\'t sort) 1 (do) or 2 (heuristic)\n\nspread_kerevalmeth :: Cint\n\npassed to spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)\n\nspread_kerpad :: Cint\n\npassed to spread_opts, 0: don\'t pad to mult of 4, 1: do\n\nchkbnds :: Cint\n\n0: don\'t check if input NU pts in [-3pi,3pi], 1: do\n\nfftw :: Cint\n\n0:FFTW_ESTIMATE, or 1:FFTW_MEASURE (slow plan but faster)\n\nmodeord :: Cint\n\n0: CMCL-style increasing mode ordering (neg to pos), or\n1: FFT-style mode ordering (affects type-1,2 only)\n\nupsampfac::Cdouble\n\nupsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)\n\n\n\n\n\n"
},

{
    "location": "#Types-1",
    "page": "FINUFFT.jl Reference",
    "title": "Types",
    "category": "section",
    "text": "Modules = [FINUFFT]\nOrder = [:type]"
},

{
    "location": "#FINUFFT.finufft_default_opts-Tuple{}",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.finufft_default_opts",
    "category": "method",
    "text": "finufft_default_opts()\n\nReturn a nufft_opts struct with the default FINUFFT settings.\nSee: https://finufft.readthedocs.io/en/latest/usage.html#options\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft1d1",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft1d1",
    "category": "function",
    "text": "nufft1d1(xj      :: Array{Float64}, \n         cj      :: Array{ComplexF64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         ms      :: Integer\n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-1 1D complex nonuniform FFT. \n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft1d1!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft1d1!",
    "category": "function",
    "text": "nufft1d1!(xj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          fk      :: Array{ComplexF64} \n          [, opts :: nufft_opts]\n        )\n\nCompute type-1 1D complex nonuniform FFT. Output stored in fk.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft1d2",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft1d2",
    "category": "function",
    "text": "nufft1d2(xj      :: Array{Float64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         fk      :: Array{ComplexF64} \n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-2 1D complex nonuniform FFT. \n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft1d2!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft1d2!",
    "category": "function",
    "text": "nufft1d2!(xj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          fk      :: Array{ComplexF64} \n          [, opts :: nufft_opts]\n        )\n\nCompute type-2 1D complex nonuniform FFT. Output stored in cj.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft1d3",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft1d3",
    "category": "function",
    "text": "nufft1d3(xj      :: Array{Float64}, \n         cj      :: Array{ComplexF64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         sk      :: Array{Float64},\n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-3 1D complex nonuniform FFT.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft1d3!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft1d3!",
    "category": "function",
    "text": "nufft1d3!(xj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          sk      :: Array{Float64},\n          fk      :: Array{ComplexF64},\n          [, opts :: nufft_opts]\n         )\n\nCompute type-3 1D complex nonuniform FFT. Output stored in fk.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft2d1",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft2d1",
    "category": "function",
    "text": "nufft2d1(xj      :: Array{Float64}, \n         yj      :: Array{Float64}, \n         cj      :: Array{ComplexF64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         ms      :: Integer,\n         mt      :: Integer,\n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-1 2D complex nonuniform FFT.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft2d1!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft2d1!",
    "category": "function",
    "text": "nufft2d1!(xj      :: Array{Float64}, \n          yj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          fk      :: Array{ComplexF64} \n          [, opts :: nufft_opts]\n        )\n\nCompute type-1 2D complex nonuniform FFT. Output stored in fk.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft2d2",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft2d2",
    "category": "function",
    "text": "nufft2d2(xj      :: Array{Float64}, \n         yj      :: Array{Float64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         fk      :: Array{ComplexF64} \n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-2 2D complex nonuniform FFT. \n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft2d2!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft2d2!",
    "category": "function",
    "text": "nufft2d2!(xj      :: Array{Float64}, \n          yj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          fk      :: Array{ComplexF64} \n          [, opts :: nufft_opts]\n        )\n\nCompute type-2 2D complex nonuniform FFT. Output stored in cj.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft2d3",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft2d3",
    "category": "function",
    "text": "nufft2d3(xj      :: Array{Float64}, \n         yj      :: Array{Float64},\n         cj      :: Array{ComplexF64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         sk      :: Array{Float64},\n         tk      :: Array{Float64}\n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-3 2D complex nonuniform FFT.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft2d3!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft2d3!",
    "category": "function",
    "text": "nufft2d3!(xj      :: Array{Float64}, \n          yj      :: Array{Float64},\n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          sk      :: Array{Float64},\n          tk      :: Array{Float64},\n          fk      :: Array{ComplexF64}\n          [, opts :: nufft_opts]\n         )\n\nCompute type-3 2D complex nonuniform FFT. Output stored in fk.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft3d1",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft3d1",
    "category": "function",
    "text": "nufft3d1(xj      :: Array{Float64}, \n         yj      :: Array{Float64}, \n         zj      :: Array{Float64}, \n         cj      :: Array{ComplexF64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         ms      :: Integer,\n         mt      :: Integer,\n         mu      :: Integer,\n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-1 3D complex nonuniform FFT.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft3d1!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft3d1!",
    "category": "function",
    "text": "nufft3d1!(xj      :: Array{Float64}, \n          yj      :: Array{Float64}, \n          zj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          fk      :: Array{ComplexF64} \n          [, opts :: nufft_opts]\n        )\n\nCompute type-1 3D complex nonuniform FFT. Output stored in fk.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft3d2",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft3d2",
    "category": "function",
    "text": "nufft3d2(xj      :: Array{Float64}, \n         yj      :: Array{Float64}, \n         zj      :: Array{Float64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         fk      :: Array{ComplexF64} \n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-2 3D complex nonuniform FFT. \n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft3d2!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft3d2!",
    "category": "function",
    "text": "nufft3d2!(xj      :: Array{Float64}, \n          yj      :: Array{Float64}, \n          zj      :: Array{Float64}, \n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          fk      :: Array{ComplexF64} \n          [, opts :: nufft_opts]\n        )\n\nCompute type-2 3D complex nonuniform FFT. Output stored in cj.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft3d3",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft3d3",
    "category": "function",
    "text": "nufft3d3(xj      :: Array{Float64}, \n         yj      :: Array{Float64},\n         zj      :: Array{Float64},\n         cj      :: Array{ComplexF64}, \n         iflag   :: Integer, \n         eps     :: Float64,\n         sk      :: Array{Float64},\n         tk      :: Array{Float64}\n         uk      :: Array{Float64}\n         [, opts :: nufft_opts]\n        ) -> Array{ComplexF64}\n\nCompute type-3 3D complex nonuniform FFT.\n\n\n\n\n\n"
},

{
    "location": "#FINUFFT.nufft3d3!",
    "page": "FINUFFT.jl Reference",
    "title": "FINUFFT.nufft3d3!",
    "category": "function",
    "text": "nufft3d3!(xj      :: Array{Float64}, \n          yj      :: Array{Float64},\n          zj      :: Array{Float64},\n          cj      :: Array{ComplexF64}, \n          iflag   :: Integer, \n          eps     :: Float64,\n          sk      :: Array{Float64},\n          tk      :: Array{Float64},\n          uk      :: Array{Float64},\n          fk      :: Array{ComplexF64}\n          [, opts :: nufft_opts]\n         )\n\nCompute type-3 3D complex nonuniform FFT. Output stored in fk.\n\n\n\n\n\n"
},

{
    "location": "#Functions-1",
    "page": "FINUFFT.jl Reference",
    "title": "Functions",
    "category": "section",
    "text": "Modules = [FINUFFT]\nOrder = [:function]"
},

]}
