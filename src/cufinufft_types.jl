const cufinufft_plan_c = Ptr{Cvoid}

mutable struct cufinufft_plan{T}
    type       :: Cint
    ntrans     :: Cint
    dim        :: Cint
    ms         :: Int64
    mt         :: Int64
    mu         :: Int64
    nj         :: Int64
    nk         :: Int64
    plan_ptr   :: cufinufft_plan_c
    # Arrays used for keeping references to input data alive.
    # These should not be modified directly, as it will have no
    # effect.
    _x_d       :: CuVector{T}
    _y_d       :: CuVector{T}
    _z_d       :: CuVector{T}
    _s_d       :: CuVector{T}
    _t_d       :: CuVector{T}
    _u_d       :: CuVector{T}
    # Default constructor that does not require input arrays
    cufinufft_plan{T}(type, ntrans, dim, ms, mt, mu, nj, nk, plan_ptr) where T <: finufftReal =
        new(type, ntrans, dim, ms, mt, mu, nj, nk, plan_ptr, T[], T[], T[], T[], T[], T[])
end

# This mus match definition in
# finufft/include/cufinufft_opts.h
mutable struct cufinufft_opts
    upsampfac            :: Cdouble # upsampling ratio sigma, only 2.0 (standard) is implemented

    # following options are for gpu #
    gpu_method           :: Cint # 1: nonuniform-pts driven, 2: shared mem (SM)
    gpu_sort             :: Cint # when NU-pts driven: 0: no sort (GM), 1: sort (GM-sort)

    gpu_binsizex         :: Cint # used for 2D, 3D subproblem method
    gpu_binsizey         :: Cint
    gpu_binsizez         :: Cint

    gpu_obinsizex        :: Cint # used for 3D spread block gather method
    gpu_obinsizey        :: Cint
    gpu_obinsizez        :: Cint

    gpu_maxsubprobsize   :: Cint
    gpu_kerevalmeth      :: Cint # 0: direct exp(sqrt()), 1: Horner ppval

    gpu_spreadinterponly :: Cint # 0: NUFFT, 1: spread or interpolation only

    gpu_maxbatchsize     :: Cint

    # multi-gpu support #
    gpu_device_id        :: Cint

    gpu_stream           :: Ptr{Cvoid}

    modeord              :: Cint # (type 1,2 only): 0 CMCL-style increasing mode order
                                 #                  1 FFT-style mode order
    cufinufft_opts() = new()
end
