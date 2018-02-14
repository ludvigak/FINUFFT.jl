__precompile__()

module FINUFFT
using PyCall

nufft1d1! = PyNULL()
nufft1d2! = PyNULL()
nufft1d3! = PyNULL()
nufft2d1! = PyNULL()
nufft2d2! = PyNULL()
nufft2d3! = PyNULL()
nufft3d1! = PyNULL()
nufft3d2! = PyNULL()
nufft3d3! = PyNULL()

function __init__()
    finufftpy = pyimport("finufftpy")

    copy!(nufft1d1!, finufftpy[:nufft1d1])
    copy!(nufft1d2!, finufftpy[:nufft1d2])
    copy!(nufft1d3!, finufftpy[:nufft1d3])    
    
    copy!(nufft2d1!, finufftpy[:nufft2d1])
    copy!(nufft2d2!, finufftpy[:nufft2d2])
    copy!(nufft2d3!, finufftpy[:nufft2d3])

    copy!(nufft3d1!, finufftpy[:nufft3d1])
    copy!(nufft3d2!, finufftpy[:nufft3d2])
    copy!(nufft3d3!, finufftpy[:nufft3d3])        
end

export nufft1d1!, nufft1d2!, nufft1d3!
export nufft2d1!, nufft2d2!, nufft2d3!
export nufft3d1!, nufft3d2!,  nufft3d3!

end # module
