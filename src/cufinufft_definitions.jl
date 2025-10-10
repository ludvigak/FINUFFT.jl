export cufinufft_makeplan
export cufinufft_destroy!
export cufinufft_setpts!
export cufinufft_exec
export cufinufft_exec!

# We define the plan here so that we can dispatch on it in other packages
# Because CUDA may not be loaded when this file is included we set the arrays as abstract
const cufinufft_plan_c = Ptr{Cvoid}
mutable struct cufinufft_plan{T, V<:AbstractVector{T}}
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
    _x_d       :: V
    _y_d       :: V
    _z_d       :: V
    _s_d       :: V
    _t_d       :: V
    _u_d       :: V
end


using cufinufft_jll

function cufinufft_makeplan end
function cufinufft_destroy! end
function cufinufft_setpts! end
function cufinufft_exec end
function cufinufft_exec! end
function cufinufft_default_opts end
