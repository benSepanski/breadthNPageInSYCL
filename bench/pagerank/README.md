# Tuxedo Instructions

## Lonestar GPU Implementations

Full documentation is at [Galois/Lonestar](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/pagerank).
To run bfs using load-balancing on `rmat15` starting at node `0` and writing
the results to `pagerank-rmat15.txt`, run
```bash
$GALOIS_BUILD_DIR/lonestar/analytics/gpu/pagerank/pagerank-gpu -o pagerank-rmat15.txt /net/ohm/export/iss/dist-inputs/rmat15.gr
```
Passing `-g` allows you to specify the `CUDA_DEVICE` (default 0) (see [skelapp in libgpu](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/src/skelapp/skel.cu)).
