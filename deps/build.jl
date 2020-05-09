using BinDeps
using Libdl
using Conda

@BinDeps.setup

## First install fftw
fftw = library_dependency("libfftw3", aliases = ["fftw3"])

if !Sys.iswindows()
    fftw_threads = library_dependency("libfftw3_threads")
end

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
if !Sys.iswindows()
    provides(BuildProcess,
         (@build_steps begin
          end),
         fftw_threads)
end
# ...because this does not work:
#provides(Binaries, "usr", fftw)
#provides(Binaries, "usr", fftw_threads)

if Sys.iswindows()
    try
        @BinDeps.install Dict(
            :libfftw3 => :fftw
        )
    catch
        # fftw3 has not been found
        provides(Binaries, joinpath(pwd(),"usr","Library","bin"), fftw)
        @BinDeps.install Dict(
            :libfftw3 => :fftw
        )
    end
else
    @BinDeps.install Dict(
        :libfftw3 => :fftw,
        :libfftw3_threads => :fftw_threads
    )
end

## Now fftw is in place, download and build finufft
libfinufft = library_dependency("libfinufft")

provides(Sources,
        URI("https://github.com/flatironinstitute/finufft/archive/master.zip"),
         libfinufft,
         unpacked_dir = "finufft-master")

rootdir = joinpath(BinDeps.srcdir(libfinufft), "finufft-master")
libname = "libfinufft." * Libdl.dlext
libfile = joinpath(BinDeps.libdir(libfinufft),libname)
buildfile = joinpath(rootdir, "lib", libname)

if Sys.iswindows()
    @show lib = joinpath(usr, "Library", "lib")
    @show inc = joinpath(usr, "Library", "include")
else
    @show lib = BinDeps.libdir(fftw)
    @show inc = BinDeps.includedir(fftw)
end

# "FFTWOMPSUFFIX=threads" because Conda doesn't supply libfftw3_omp 
if Sys.KERNEL == :Darwin
    buildcmd = `make lib/libfinufft.so LIBRARY_PATH=$lib CPATH=$inc FFTWOMPSUFFIX=threads CXX=g++-9 CC=gcc-9`
elseif Sys.iswindows()
    buildcmd = `make lib OMP=OFF LIBRARY_PATH=$lib CPATH=$inc DYNAMICLIB=lib/$libname`
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
            # `cp $buildfile $libfile`
        end
    end
run(finufftbuild)
cp(buildfile, libfile, force=true)

# Just add to deps.jl to bypass BinDeps checking
depsfile_location = joinpath(splitdir(Base.source_path())[1],"deps.jl")
fh = open(depsfile_location, "a")
Sys.iswindows() ? write(fh, "\n@checked_lib libfinufft \"$(replace(libfile, "\\" => "\\\\"))\"\n") : write(fh, "\n@checked_lib libfinufft \"$libfile\"\n")
close(fh)

