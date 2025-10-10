module CUFINUFFTExt

if isdefined(Base, :get_extension)
    using FINUFFT
    using FINUFFT: cufinufft_plan
else
    using ..FINUFFT
    using ..FINUFFT: cufinufft_plan
end
using CUDA

include("cufinufft.jl")
include("cufinufft_simple.jl")

function __init__()
    determine_cuda_status()
end



end