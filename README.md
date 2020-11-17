# breadthNPageInSYCL

TODO: SET UP SCRIPT TO RUN THE LONESTAR APPLICATIONS ON GRAPHS

TODO: MAKE SURE SCRIPT RECORDS METRICS OF INTEREST (TIMING, KERNEL THROUGHPUT, ETC.)

TODO: WRITE SAMPLE SYCL PAGERANK (RIGHT NOW IT IS HELLO WORLDS)

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
* `libsyclutils` provides a skeleton for the bfs and pagerank
  SYCL applications, as well as some objects needed in both
  (such as a graph which plays with SYCL,
  and a worklist)

## SYCL Resources

The [manual](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf)
is a surprisingly good resource.
You should probably look at one or two hello-world
SYCL programs first though (e.g. [this tutorial](https://tech.io/playgrounds/48226/introduction-to-sycl/getting-started))

* A good resource for the SYCL computing paradigms is chapter 3 of the manual.
  The chapter is also short enough that you can read through it
  (especially if you skip over the OpenCL sections)

    - Section 3.5.2 has some good information on the SYCL memory model.
      Section 3.7 explicitly describes the memory objects. 

    - Section 3.6.3 describes a convenient syntax for representing
      tasks that have an outer loop distributed across work groups
      and inner loop distributed across work items!

    - To perform outlining (as described in section 4.1
      of the [IrGL Paper](https://dl.acm.org/doi/10.1145/2983990.2984015)
      we will probably use the `single_task` as described
      in section 3.6.4 of the manual

    - Synchronization is described in section 3.6.5.2

* Chapter 4 of the manual covers specific functions. It is a good 
  reference for classes,
  but way too long to read through.
  Use the table of contents to jump to specific classes you are
  interested in.

    - [here](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf#page=150&zoom=100,96,730)
      is the documentation for items/ranges/etc.

    - explicit memory management is discussed [here](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf#page=182&zoom=100,96,329)
      in the manual

[Here](https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling) is
a nice error-handling guide. Make sure you put also try putting a try/catch
block around the first `queue.submit` call.

Importantly: SYCL atomics don't work with 64-bit integers on NVIDIA gpus yet.

## Setup & Installation

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

First clone this repository, set `$SOURCE_DIR`
to the root directory of the repository, and set
`$BUILD_DIR` to whichever directory you want to build
the project in.
```
git clone https://github.com/benSepanski/breadthNPageInSYCL
cd breadthNPageInSYCL
SOURCE_DIR=`pwd`
BUILD_DIR=$SOURCE_DIR/build  # or wherever you want
```

We have to explicitly set several options.
The sample code below has the correct values for the Tuxedo machine
(when set up as described in [Tuxedo Setup](#tuxedo-setup)).
Modify the following CMake variables as necessary for your machine:

* `ComputeCpp_DIR` The directory of your `ComputeCpp` build
* `OpenCL_INCLUDE_DIR` A directory holding the OpenCL headers.
  Look at the environment variable `$OpenCL_INCLUDE_DIR` to
  see what this should be.
* `COMPUTECPP_USER_FLAGS` Used to specify the gcc version for ComputeCpp to use.
* `COMPUTECPP_BITCODE` Specifies the bitcode for ComputeCpp to target.
   `"ptx64"` for NVIDIA as described
   [here](https://developer.codeplay.com/products/computecpp/ce/guides/platform-support/targeting-nvidia-ptx).
* `CMAKE_C_COMPILER` the c compiler to use
* `CMAKE_CXX_COMPILER` the c++ compiler to use
* `CL_TARGET_OPENCL_VERSION` is a 3-digit number representing the OpenCL version
  (e.g. 300 represents 3.0.0).
  Assuming the environment variable `$OpenCL_LIBRARY` is set,
  you can find the OpenCL version by looking at `$OpenCL_LIBRARY/pkgconfig/OpenCL.pc`
* `GALOIS_CUDA_CAPABILITY` Cuda capability used in Galois build.
  For tuxedo this is "3.7;6.1". This is used when
  building LonestarGPU as described [here](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu)
* `CMAKE_BUILD_TYPE` `"Release"` or `"Debug"`.

Assuming the source directory (i.e. where this `README.md` file is located)
is in `$SOURCE_DIR` and you want to build into directory `$BUILD_DIR`
(The Galois build directory will be `$BUILD_DIR/extern/Galois`), run
```bash
mkdir -p $BUILD_DIR
cmake -S $SOURCE_DIR -B $BUILD_DIR \
    -DComputeCpp_DIR="/org/centers/cdgc/ComputeCpp/ComputeCpp-CE-2.2.1-x86_64-linux-gnu/" \
    -DOpenCL_INCLUDE_DIR="/org/centers/cdgc/ComputeCpp/OpenCL-Headers" \
    -DCOMPUTECPP_USER_FLAGS="--gcc-toolchain=/opt/apps/ossw/applications/gcc/gcc-8.1/c7" \
    -DCOMPUTECPP_BITCODE="ptx64" \
    -DCMAKE_C_COMPILER=`which gcc` \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DCL_TARGET_OPENCL_VERSION=300 \
    -DGALOIS_CUDA_CAPABILITY="3.7;6.1" \
    -DCMAKE_BUILD_TYPE="Release"
```

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
