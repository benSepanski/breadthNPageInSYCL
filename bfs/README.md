# BFS in SYCL

TODO : Set up CMakeLists to automate compilation

To compile `helloWorld.cpp`, run 
```bash
compute++ -sycl -sycl-target ptx64 --gcc-toolchain=/opt/apps/ossw/applications/gcc/gcc-8.1/c7 -c helloWorld.cpp
```
The `-sycl` flag tells `compute++` to compile `sycl`, the `-sycl-target ptx64`
tells it to target NVIDIA machines as described [here](https://developer.codeplay.com/products/computecpp/ce/guides/platform-support/targeting-nvidia-ptx).
The `--gcc-toolchain` argument points it towards the right `gcc`.
