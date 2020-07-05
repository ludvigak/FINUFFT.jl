using BinDeps
using Libdl
using Conda


@BinDeps.setup

if haskey(ENV, "JULIA_FFTW_PROVIDER")
    const provider = ENV["JULIA_FFTW_PROVIDER"]
else
    const provider = "FFTW"
end

## First install fftw
if provider == "FFTW"
    if Sys.iswindows()
        fftw = library_dependency("fftw3")
    else
        fftw = library_dependency("libfftw3")
    end
elseif provider == "MKL"
    if Sys.iswindows()
        fftw = library_dependency("mkl_rt")
    else
        fftw = library_dependency("libmkl_rt")
    end
end

if !Sys.iswindows()
    fftw_threads = library_dependency("libfftw3_threads")
end

# Conda just dumps the binaries into the right place
usr = usrdir(fftw)
if !isdir(usr)
    mkdir(usr)
end

# but this place depends on the OS
if Sys.iswindows()
    dependency_path = joinpath(usr,"Library")
else
    dependency_path = usr
end


if provider == "FFTW"
    Conda.add("fftw", usr)
else
    Conda.add("mkl_fft", usr)
    Conda.add("mkl-include", usr)
end


if Sys.iswindows()
    provides(Binaries, joinpath(dependency_path, "bin"), fftw)
else
    provides(Binaries, joinpath(dependency_path, "lib"), fftw)
    provides(Binaries, joinpath(dependency_path, "lib"), fftw_threads)
end


if provider == "FFTW"
    if Sys.iswindows()
        @BinDeps.install Dict(
            :fftw3 => :fftw
        )
    else
        @BinDeps.install Dict(
            :libfftw3 => :fftw,
            :libfftw3_threads => :fftw_threads
        )
    end
elseif provider == "MKL"
    if Sys.iswindows()
        @BinDeps.install Dict(
                :mkl_rt => :fftw
            )
    else
        @BinDeps.install Dict(
                :libmkl_rt => :fftw
            )
    end
end

## Now fftw is in place, download and build finufft
libfinufft = library_dependency("libfinufft")

provides(Sources,
        URI("https://github.com/flatironinstitute/finufft/archive/v1.1.2.zip"),
         libfinufft,
         unpacked_dir = "finufft-1.1.2")

rootdir = joinpath(BinDeps.srcdir(libfinufft), "finufft-1.1.2")
libname = "libfinufft.so"
libfile = joinpath(BinDeps.libdir(libfinufft),libname)
buildfile = joinpath(rootdir, "lib", libname)

if Sys.iswindows()
    if provider == "FFTW"
        @show lib = joinpath(usr, "Library", "lib")
        @show inc = joinpath(usr, "Library", "include")
    else
        @show lib = joinpath(usr, "Library", "bin")
        @show inc = joinpath(usr, "Library", "include", "fftw")
    end
else
    @show lib = BinDeps.libdir(fftw)
    @show inc = BinDeps.includedir(fftw)
end

# "FFTWOMPSUFFIX=threads" because Conda doesn't supply libfftw3_omp 
if Sys.KERNEL == :Darwin
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc FFTWOMPSUFFIX=threads CXX=g++-9 CC=gcc-9`
elseif Sys.iswindows()
    buildcmd = `make lib OMP=OFF LIBRARY_PATH=$lib CPATH=$inc FFTWNAME=$(fftw.name)`
else
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc FFTWNAME=$(fftw.name) FFTWOMPSUFFIX=threads $(provider == "MKL" ? "OMP=OFF" : "")`
end

finufftbuild = 
    @build_steps begin
        GetSources(libfinufft)
        @build_steps begin
            ChangeDirectory(rootdir)
            buildcmd
            CreateDirectory(libdir(libfinufft))
        end
    end
run(finufftbuild)
cp(buildfile, libfile, force=true)

# Just add to deps.jl to bypass BinDeps checking
depsfile_location = joinpath(splitdir(Base.source_path())[1],"deps.jl")
fh = open(depsfile_location, "a")
if Sys.iswindows()
    libfile = replace(libfile, "\\" => "\\\\")
end
write(fh, "\n@checked_lib libfinufft \"$libfile\"\n")
close(fh)

