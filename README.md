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

## Installation

This section describes how to build our SYCL implementations of
bfs and pagerank.

If you are using the Tuxedo machine, go through [Tuxedo Setup](#tuxedo-setup)
first. Even if you are not using the Tuxedo machine, it may be worth
looking at the [Connecting to Tuxedo](#connecting-to-tuxedo) portion
to see the versions of packages we are using.

Assuming the source directory (i.e. where this `README.md` file is located)
is in `$SOURCE_DIR` and you want to build into directory `$BUILD_DIR`, run
```bash
mkdir -p $BUILD_DIR
cmake -S $SOURCE_DIR -B $BUILD_DIR # <CMAKE OPTIONS HERE> 
                                   # -DComputeCpp_DIR=...
                                   # -DOpenCL_INCLUDE_DIR=...
                                   # <etcetera>
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
you do any of the `module load`s described in [Connecting to Tuxedo](#connecting-to-tuxedo).

## Tuxedo Setup

### Connecting to Tuxedo

1. If you are not on campus, connect to the [UT VPN](https://wikis.utexas.edu/display/engritgpublic/Connecting+to+the+University+of+Texas+VPN).
2. ssh into the machine by running `ssh <yourUsername>@tuxedo.oden.utexas.edu`.
3. To see available NVIDIA GPUs, run `nvidia-smi`. You should see four Tesla K80s and two GeForce GTX 1080s.
4. Installing Galois will depend on several modern compilers/libraries. 
   We will also need to be able to run cuda and SYCL code. We need to make sure
   the proper modules are loaded by running the following commands:

```bash
module use /org/centers/cdgc/modules # Make sure we can see all the modules we will need:
module use /net/faraday/workspace/local/modules/modulefiles # Make sure we can see all the modules we will need:
# Now get the right versions needed for Galois
module load c7 gcc/8.1 cmake/3.17.0 boost/1.71 llvm/10.0 fmt/6.2.1 mpich2/3.2
# Get version of cuda and get computeCpp SYCL compiler
module load cuda/10.2 compute-cpp/2.2.1
# useful so that git doesn't yell at you
module load git/2.14.2
```
You'll need to load these modules every time you log onto the machine.

### Lonestar

[Galois](https://iss.oden.utexas.edu/?p=projects/galois) is a project which exploits irregular
parallelism in code. Installation instructions and source code can be found
on the project [github repo](https://github.com/IntelligentSoftwareSystems/Galois).
The [Galois documentation](https://iss.oden.utexas.edu/projects/galois/api/current/index.html) has
a [tutorial](https://iss.oden.utexas.edu/projects/galois/api/current/tutorial.html).
We will be using release 6.0 for comparison.

It also contains the [Lonestar Project](https://iss.oden.utexas.edu/?p=projects/galois/lonestar)
(and [LonestarGPU](https://iss.oden.utexas.edu/?p=projects/galois/lonestargpu))
from which we obtain competing implementations of BFS and PR.
To install Lonestar we will follow the instructions from the Galois repository
to install the BFS and PR implementations from release 6.0:
```bash
git clone -b release-6.0 https://github.com/IntelligentSoftwareSystems/Galois
SRC_DIR=~/Galois
```
For the Lonestar [GPU benchmarks](https://github.com/IntelligentSoftwareSystems/Galois),
we need some extra dependencies
```bash
cd $SRC_DIR
git submodule init
git submodule update
```
Now make a build directory and build Galois
```bash
BUILD_DIR=$SRC_DIR/build
mkdir -p $BUILD_DIR
# The CUDA versions are for a GTX 1080 and a K80
cmake -S $SRC_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DGALOIS_CUDA_CAPABILITY="3.7;6.1"
# Build graph conversion capabilities
make graph-convert
```
Note that on our machine, `HUGE_PAGES` is on and libnuma.so is linked.

Next, build the BFS and PR applications
```bash
for application in bfs pagerank ; do
    make -C $BUILD_DIR/lonestar/analytics/cpu/$application -j
    make -C $BUILD_DIR/lonestar/analytics/gpu/$application -j
done
```
Now the bfs cpu
executable is the file `$BUILD_DIR/lonestar/analytics/cpu/bfs/bfs-cpu`, etc.

There are instructions for running the executables on the github:
* [bfs cpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/cpu/bfs)
* [bfs gpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/bfs)
* [pagerank cpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/cpu/pagerank)
* [pagerank gpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/pagerank)
