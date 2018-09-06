using BinDeps
using Compat.Libdl

@BinDeps.setup

libfinufft = library_dependency("libfinufft")

provides(Sources,
         URI("https://github.com/flatironinstitute/finufft/archive/v1.0.zip"),
         libfinufft,
         unpacked_dir = "finufft-1.0")

rootdir = joinpath(BinDeps.srcdir(libfinufft), "finufft-1.0")
libname = "libfinufft." * Libdl.dlext
libfile = joinpath(BinDeps.libdir(libfinufft),libname)
buildfile = joinpath(rootdir, "lib", libname)

buildcmd = `make lib`

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

@BinDeps.install Dict(:libfinufft => :libfinufft)
