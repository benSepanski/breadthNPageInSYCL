# breadthNPageInSYCL

TODO: SET UP SCRIPT TO RUN THE LONESTAR APPLICATIONS ON GRAPHS

TODO: MAKE SURE SCRIPT RECORDS METRICS OF INTEREST (TIMING, KERNEL THROUGHPUT, ETC.)

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
