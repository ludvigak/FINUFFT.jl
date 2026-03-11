module CUFINUFFTExt

if isdefined(Base, :get_extension)
    using FINUFFT
    using FINUFFT: cufinufft_plan, cufinufft_plan_c
else
    using ..FINUFFT
    using ..FINUFFT: cufinufft_plan, cufinufft_plan_c
end
using CUDA

include("cufinufft.jl")
include("cufinufft_simple.jl")

function __init__()
    determine_cuda_status()
end



end