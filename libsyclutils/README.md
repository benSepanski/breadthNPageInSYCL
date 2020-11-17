# SYCL Utils

This library is based on
[libgpu](https://github.com/IntelligentSoftwareSystems/Galois/tree/master/libgpu)
in Galois. Large sections of code are directly copied and modified
from libgpu.

## Directory Structure

* `src/sycl_driver.cpp` The application driver
* `include/host_csr_graph.h` A CSR graph on the host
* `include/sycl_csr_graph.h` A CSR graph represented as SYCL buffers
* `include/nvidia_selector.h` and `src/nvidia_selector.h` implement
  SYCL device selectors which can select NVIDIA GPUs from NVIDIA
  IDs
