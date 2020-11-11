# Lonestar CUDA Implementation Notes

Each of the directories contains a `README.md`
describing the given Lonestar CUDA implementation

## What is gg\_main?

One should note that ["gg.h"](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/include/gg.h)
and ["ggcuda.h"](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/include/ggcuda.h)
are from [libgpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/libgpu).

On libgpu you can see how [CSR on the GPU](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/src/csr_graph.cu) is implemented.
Moreover, you can see a [skeleton application](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/src/skelapp/skel.cu)
which shows the `main()` function used, as well
as how command line arguments are parsed.
This explains the `gg_main()` function which you see in
`bfs.cu` and `pagerank.cu`.
One should note that the first CSR argument to `gg_main()` is
the graph on the host, and the second CSR argument
to `gg_main()` is the graph on the device.
