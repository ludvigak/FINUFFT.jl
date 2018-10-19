using BinDeps
using Compat.Libdl
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
         URI("https://github.com/flatironinstitute/finufft/archive/v1.0.zip"),
         libfinufft,
         unpacked_dir = "finufft-1.0")

rootdir = joinpath(BinDeps.srcdir(libfinufft), "finufft-1.0")
libname = "libfinufft." * Libdl.dlext
libfile = joinpath(BinDeps.libdir(libfinufft),libname)
buildfile = joinpath(rootdir, "lib", "libfinufft.so")

@show lib = BinDeps.libdir(fftw)
@show inc = BinDeps.includedir(fftw)

if Sys.KERNEL == :Darwin
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc CXX=g++-8 CC=gcc-8`
else
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc`
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

