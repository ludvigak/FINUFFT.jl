__precompile__()
module FINUFFT

using Requires

## Export
export nufft1d1, nufft1d2, nufft1d3
export nufft2d1, nufft2d2, nufft2d3
export nufft3d1, nufft3d2, nufft3d3

export nufft1d1!, nufft1d2!, nufft1d3!
export nufft2d1!, nufft2d2!, nufft2d3!
export nufft3d1!, nufft3d2!, nufft3d3!

export nufft_opts
export nufft_c_opts        # backward-compatibility - remove?
export finufft_plan
export finufft_default_opts
export finufft_makeplan
export finufft_setpts!
export finufft_exec
export finufft_destroy!
export finufft_exec!
export BIGINT
export finufftReal

# By default we depend on our precompiled generic binary package...
using finufft_jll
const libfinufft = finufft_jll.libfinufft
#
# If instead you want to use your locally-compiled FINUFFT library for more
# performance, comment out the above two code lines, uncomment the upcoming
# one, and edit it for the location of your local FINUFFT installation. You
# then need to use this FINUFFT.jl pkg in dev mode and restart (see README.md):
#const libfinufft = "/PATH/TO/YOUR/finufft/lib/libfinufft.so"

# Includes
include("types.jl")
include("errors.jl")
include("helpers.jl")

# Finally, bring in the main user-facing interfaces...
include("guru.jl")
include("simple.jl")

# Only load cuFINUFFT interface if CUDA is present (via `using CUDA`) and functional
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using .CUDA
        if CUDA.functional()
            include("cufinufft.jl")
        end
    end
end

end # module
