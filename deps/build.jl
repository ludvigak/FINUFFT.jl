using BinDeps
using Libdl
using Conda

@BinDeps.setup

## First install fftw
fftw = library_dependency("libfftw3")
fftw_threads = library_dependency("libfftw3_threads")

# Conda just dumps the binaries into the right place
usr = usrdir(fftw)
if !isdir(usr)
    mkdir(usr)
end
Conda.add("fftw", usr)

# Dummy build processes
provides(BuildProcess,
         (@build_steps begin
          end),
         fftw)
provides(BuildProcess,
         (@build_steps begin
          end),
         fftw_threads)
# ...because this does not work:
#provides(Binaries, "usr", fftw)
#provides(Binaries, "usr", fftw_threads)

@BinDeps.install Dict(
    :libfftw3 => :fftw,
    :libfftw3_threads => :fftw_threads
)

## Now fftw is in place, download and build finufft
libfinufft = library_dependency("libfinufft")

provides(Sources,
         URI("https://github.com/flatironinstitute/finufft/archive/1.1.1.zip"),
         libfinufft,
         unpacked_dir = "finufft-1.1.1")

rootdir = joinpath(BinDeps.srcdir(libfinufft), "finufft-1.1.1")
libname = "libfinufft." * Libdl.dlext
libfile = joinpath(BinDeps.libdir(libfinufft),libname)
buildfile = joinpath(rootdir, "lib", "libfinufft.so")

@show lib = BinDeps.libdir(fftw)
@show inc = BinDeps.includedir(fftw)

# "FFTWOMPSUFFIX=threads" because Conda doesn't supply libfftw3_omp 
if Sys.KERNEL == :Darwin
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc FFTWOMPSUFFIX=threads CXX=g++-9 CC=gcc-9`
else
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc FFTWOMPSUFFIX=threads`
end

finufftbuild = 
    @build_steps begin
        GetSources(libfinufft)
        @build_steps begin
            ChangeDirectory(rootdir)
            buildcmd
            CreateDirectory(libdir(libfinufft))
            `cp $buildfile $libfile`
        end
    end
run(finufftbuild)

# Just add to deps.jl to bypass BinDeps checking
depsfile_location = joinpath(splitdir(Base.source_path())[1],"deps.jl")
fh = open(depsfile_location, "a")
write(fh, "\n@checked_lib libfinufft \"$libfile\"\n")
close(fh)

