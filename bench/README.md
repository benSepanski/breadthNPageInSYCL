# Benchmarks

* In the future, we will add more graphs of interest
  and add more automation.
  For now, we provide instructions for benchmarking
  `rmat15` on Tuxedo.

* Make sure `$GALOIS_BUILD_DIR` holds the build
  directory of Galois

## Tuxedo Instructions

### Lonestar GPU Implementations

#### BFS

Full documentaiton is at [Galois/Lonestar](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/lonestar/analytics/gpu/bfs).
To run bfs using load-balancing on `rmat15` starting at node `0` and writing
the results to `bfs-rmat15.txt`, run
```bash
$GALOIS_BUILD_DIR/lonestar/analytics/gpu/bfs/bfs-gpu -l -o bfs-rmat15.txt -s 0 /net/ohm/export/iss/dist-inputs/rmat15.gr
```
