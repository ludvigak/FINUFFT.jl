module FFTWLock

using FFTW
import FINUFFT

# When FFTW.jl is loaded alongside FINUFFT, replace the default ReentrantLock
# with FFTW's own planner lock to avoid deadlocks
function __init__()
    FINUFFT.finufftlock[] = FFTW.fftwlock
end

end  # module FFTWLock
