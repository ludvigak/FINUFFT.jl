__precompile__()

module FINUFFT
using PyCall

finufft1d1! = PyNULL()
finufft1d2! = PyNULL()
finufft1d3! = PyNULL()
finufft2d1! = PyNULL()
finufft2d2! = PyNULL()
finufft2d3! = PyNULL()
finufft3d1! = PyNULL()
finufft3d2! = PyNULL()
finufft3d3! = PyNULL()

function __init__()
    finufftpy = pyimport("finufftpy")

    copy!(finufft1d1!, finufftpy[:finufft1d1])
    copy!(finufft1d2!, finufftpy[:finufft1d2])
    copy!(finufft1d3!, finufftpy[:finufft1d3])    
    
    copy!(finufft2d1!, finufftpy[:finufft2d1])
    copy!(finufft2d2!, finufftpy[:finufft2d2])
    copy!(finufft2d3!, finufftpy[:finufft2d3])

    copy!(finufft3d1!, finufftpy[:finufft3d1])
    copy!(finufft3d2!, finufftpy[:finufft3d2])
    copy!(finufft3d3!, finufftpy[:finufft3d3])        
end

export finufft1d1!, finufft1d2!, finufft1d3!
export finufft2d1!, finufft2d2!, finufft2d3!
export finufft3d1!, finufft3d2!,  finufft3d3!

end # module
