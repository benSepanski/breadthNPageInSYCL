# breadthNPageInSYCL

The goal of this repository is to provide several implementations
of breadth-first-search (BFS) and PageRank (PR),
then benchmark them on the provided machines
(Tuxedo and the Bridges Cluster) on various graphs.
We will also compare them with the [Lonestar 4.0](https://iss.oden.utexas.edu/?p=projects/galois/lonestar)
implementations of BFS and PR.
Our aim is for these benchmarks to be easily reproducible.

## Setup

### Connecting to Tuxedo

1. If you are not on campus, connect to the [UT VPN](https://wikis.utexas.edu/display/engritgpublic/Connecting+to+the+University+of+Texas+VPN).
2. ssh into the machine by running `ssh <yourUsername>@tuxedo.oden.utexas.edu`.
3. To see available NVIDIA GPUs, run `nvidia-smi`. You should see four Tesla K80s and two GeForce GTX 1080s.
4. Installing Galois will depend on several modern compilers/libraries. We need to make sure
   the proper modules are loaded by running the following commands:

```bash
module use /org/centers/cdgc/modules # Make sure we can see all the modules we will need:
module use /net/faraday/workspace/local/modules/modulefiles # Make sure we can see all the modules we will need:
# Now get the right versions needed for Galois
module load c7 gcc/8.1 cmake/3.10.2 boost/1.67.0 llvm/10.0 fmt/6.2.1
# Get version of cuda and get computeCpp SYCL compiler
module load cuda/10.2 compute-cpp/2.2.1
```

### Galois

[Galois](https://iss.oden.utexas.edu/?p=projects/galois) is a project which exploits irregular
parallelism in code. Installation instructions and source code can be found
on the project [github repo](https://github.com/IntelligentSoftwareSystems/Galois).
The [Galois documentation](https://iss.oden.utexas.edu/projects/galois/api/current/index.html) has
a [tutorial](https://iss.oden.utexas.edu/projects/galois/api/current/tutorial.html).
We will be using release 5.0 for comparison.

It also contains the [Lonestar Project](https://iss.oden.utexas.edu/?p=projects/galois/lonestar)
from which we obtain competing implementations of BFS and PR.
