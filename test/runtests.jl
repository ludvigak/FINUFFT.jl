using Test

@testset "FINUFFT" begin
    include("test_error_handler.jl")
    include("test_type_stability.jl")
    include("test_nufft.jl")
end
