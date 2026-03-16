module CUFINUFFTExt

using FINUFFT
using FINUFFT: cufinufft_plan, cufinufft_plan_c
using CUDA

include("cufinufft.jl")
include("cufinufft_simple.jl")

function __init__()
    determine_cuda_status()
end



end
