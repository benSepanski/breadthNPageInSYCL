# Tuxedo Instructions

## Lonestar GPU Implementations

Full documentaiton is at [Galois/Lonestar](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/bfs).
To run bfs using load-balancing on `rmat15` starting at node `0` and writing
the results to `bfs-rmat15.txt`, run
```bash
$GALOIS_BUILD_DIR/lonestar/analytics/gpu/bfs/bfs-gpu -l -o bfs-rmat15.txt -s 0 /net/ohm/export/iss/dist-inputs/rmat15.gr
```
Passing `-g` allows you to specify the `CUDA_DEVICE` (default 0) (see [skelapp in libgpu](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/src/skelapp/skel.cu)).
