using BinDeps
using Compat.Libdl
using Conda

@BinDeps.setup

# First install fftw
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

# Then download and build finufft
libfinufft = library_dependency("libfinufft")

provides(Sources,
         URI("https://github.com/flatironinstitute/finufft/archive/v1.0.zip"),
         libfinufft,
         unpacked_dir = "finufft-1.0")

rootdir = joinpath(BinDeps.srcdir(libfinufft), "finufft-1.0")
libname = "libfinufft." * Libdl.dlext
libfile = joinpath(BinDeps.libdir(libfinufft),libname)
buildfile = joinpath(rootdir, "lib", libname)

@show lib = BinDeps.libdir(fftw)
@show inc = BinDeps.includedir(fftw)

if Sys.KERNEL == :Darwin
    buildcmd = `make lib LIBRARY_PATH=$lib CPATH=$inc CXX=g++-4.9 CC=gcc-4.9 DYNAMICLIB=lib/libfinufft.dylib`
else
    buildcmd = `make lib LIBRARY_PATH=$lib CPATH=$inc`
end

provides(BuildProcess,
         (@build_steps begin
          GetSources(libfinufft)
          @build_steps begin
          ChangeDirectory(rootdir)
          FileRule(libfile, @build_steps begin
                   buildcmd
                   CreateDirectory(libdir(libfinufft))
                   `cp $buildfile $libfile`
                   end)
          end
          end),
         libfinufft)

@BinDeps.install Dict(
    :libfinufft => :libfinufft,
    :libfftw3 => :fftw,
    :libfftw3_threads => :fftw_threads
)
