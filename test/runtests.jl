using Test

@testset "FINUFFT" begin
    include("test_error_handler.jl")
    include("test_nufft.jl")
    include("test_thread_safety.jl")
end
@testset "cuFINUFFT" begin
    include("test_cuda.jl")
end
;
