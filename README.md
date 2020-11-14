# breadthNPageInSYCL

TODO: SET UP SCRIPT TO RUN THE LONESTAR APPLICATIONS ON GRAPHS

TODO: MAKE SURE SCRIPT RECORDS METRICS OF INTEREST (TIMING, KERNEL THROUGHPUT, ETC.)

TODO: WRITE SAMPLE SYCL BFS AND PAGERANK (RIGHT NOW THEY"RE HELLO WORLDS)

TODO: WRITE CORRECTNESS CHECK FOR BFS AND PAGERANK (MAKE SURE OUTPUT MATCHES GALOIS)

The goal of this repository is to provide several implementations
of breadth-first-search (BFS) and PageRank (PR),
then benchmark them on the provided machines
(Tuxedo and the Bridges Cluster) on various graphs.
We will also compare them with the [Lonestar](https://iss.oden.utexas.edu/?p=projects/galois/lonestar)
implementations of BFS and PR.
Our aim is for these benchmarks to be easily reproducible.

## Directory structure

* `bfs` contains the SYCL implementation of BFS
* `bench` contains instructions for benchmarking the
  implementations
* `annotatedLonestar` provides annotations on the Lonestar GPU
  implemenations of BFS and PR for quick reference

### Tuxedo Setup

1. If you are not on campus, connect to the [UT VPN](https://wikis.utexas.edu/display/engritgpublic/Connecting+to+the+University+of+Texas+VPN).
2. ssh into the machine by running `ssh <yourUsername>@tuxedo.oden.utexas.edu`.
3. To see available NVIDIA GPUs, run `nvidia-smi`. You should see four Tesla K80s and two GeForce GTX 1080s.
4. Installing Galois will depend on several modern compilers/libraries. 
   We will also need to be able to run cuda and SYCL code. We need to make sure
   the proper modules are loaded by running the following commands:

```bash
module use /org/centers/cdgc/modules # Make sure we can see all the modules we will need:
module use /net/faraday/workspace/local/modules/modulefiles # Make sure we can see all the modules we will need:
# Get the right versions needed for Galois
module load c7 gcc/8.1 cmake/3.17.0 boost/1.71 llvm/10.0 fmt/6.2.1 mpich2/3.2
# Get version of cuda and get computeCpp SYCL compiler
module load cuda/10.2 compute-cpp/2.2.1
# useful so that git doesn't yell at you
module load git/2.14.2
```
You'll need to load these modules every time you log onto the machine.

## Installation

This section describes how to build our SYCL implementations of
bfs and pagerank.

Note that we have to explicitly set the C and CXX compilers
since the default CMake files used by ComputeCpp
find the incorrect gcc on tuxedo (which messes up the build process).

Assuming the source directory (i.e. where this `README.md` file is located)
is in `$SOURCE_DIR` and you want to build into directory `$BUILD_DIR`
(The Galois build directory will be `$BUILD_DIR/extern/Galois`), run
```bash
mkdir -p $BUILD_DIR
cmake -S $SOURCE_DIR -B $BUILD_DIR \
    -DCMAKE_C_COMPILER=`which gcc` \
    -DCMAKE_CXX_COMPILER=`which g++`
```
The default cmake options are set up for the configuration of the Tuxedo machine
described in [Tuxedo Setup](#tuxedo-setup). You can look in `CMakeLists.txt`
to see the Tuxedo defaults for these options:
* `ComputeCpp_DIR` The directory of your `ComputeCpp` build
* `OpenCL_INCLUDE_DIR` A directory holding the OpenCL headers.
  Look at the environment variable `$OpenCL_INCLUDE_DIR` to
  see what this should be.
* `COMPUTECPP_USER_FLAGS` We use this to specify the gcc version to use.
  Note that our setup doesn't actually require this to be set
  since we called `module load gcc/8.1`, but we leave the option here
  for completeness.
* `COMPUTECPP_BITCODE` We set this to `"ptx64"` to tell SYCL
  to target NVIDIA machines as described
  [here](https://developer.codeplay.com/products/computecpp/ce/guides/platform-support/targeting-nvidia-ptx).
* `GALOIS_CUDA_CAPABILITY` default is "3.7;6.1", this is used when
  building LonestarGPU as described [here](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu)
* `CMAKE_BUILD_TYPE` default is `"Release"`, this is used
  when building Galois. You could build `"Debug"` instead.

Go to [bfs](https://github.com/benSepanski/breadthNPageInSYCL/tree/main/bfs)
or [pagerank](https://github.com/benSepanski/breadthNPageInSYCL/tree/main/pagerank)
directories for instructions on building the individual applications.

One note: To avoid an annoying error message when you build the applications,
make sure to set the environment variable `CL_TARGET_OPENCL_VERSION`
based on your OpenCL version.
Assuming the environment variable `$OpenCL_LIBRARY` is set,
you can find the OpenCL version by looking at `$OpenCL_LIBRARY/pkgconfig/OpenCL.pc`
For Tuxedo, set the environment variable `CL_TARGET_OPENCL_VERSION=300`
(version 3.0). This environment variable must be set before
you do any of the `module load`s described in [Connecting to Tuxedo](#tuxedo-setup).

### Galois / Lonestar

We installed [Galois](https://iss.oden.utexas.edu/?p=projects/galois)
as a submodule during cmake (source is in `$SOURCE_DIR/extern/GALOIS`,
build is in `$BUILD_DIR/extern/Galois`).

Galois is a project which exploits irregular parallelism in code. 
The [github repo](https://github.com/IntelligentSoftwareSystems/Galois) holds the source code.
The [Galois documentation](https://iss.oden.utexas.edu/projects/galois/api/current/index.html) has
a [tutorial](https://iss.oden.utexas.edu/projects/galois/api/current/tutorial.html).
We will be using release 6.0 for comparison.
Galois also contains the [Lonestar Project](https://iss.oden.utexas.edu/?p=projects/galois/lonestar)
(and [LonestarGPU](https://iss.oden.utexas.edu/?p=projects/galois/lonestargpu))
from which we obtain competing implementations of BFS and PR.

Note that on tuxedo, `HUGE_PAGES` is on and libnuma.so is linked.

You can build the BFS and PR applications by running
the following code in the Galois build directory
(`$BUILD_DIR/extern/Galois`, as described in [installation instructions](#installation))
```bash
# Run this in the Galois build directory.
for application in bfs pagerank ; do
    make -C $BUILD_DIR/extern/Galois/lonestar/analytics/cpu/$application -j
    make -C $BUILD_DIR/extern/Galois/lonestar/analytics/gpu/$application -j
done
```
Now the bfs cpu is in the `$BUILD_DIR/extern/Galois/lonestar/analytics/cpu/bfs/` directory.

There are instructions for running the executables on the github:
* [bfs cpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/cpu/bfs)
* [bfs gpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/bfs)
* [pagerank cpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/cpu/pagerank)
* [pagerank gpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/pagerank)
