module CUFINUFFTExt

if isdefined(Base, :get_extension)
    using FINUFFT
else
    using ..FINUFFT
end
using CUDA

include("cufinufft.jl")
include("cufinufft_simple.jl")

function __init__()
    determine_cuda_status()
end



end