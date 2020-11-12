# Built following the codeplay guide:
# https://developer.codeplay.com/products/computecpp/ce/guides/integration-guide

project(breadthNPageInSYCL)

cmake_minimum_required(VERSION 3.4.3)

set( ComputeCpp_DIR "/org/centers/cdgc/ComputeCpp/ComputeCpp-CE-2.2.1-x86_64-linux-gnu/" )

set( OpenCL_INCLUDE_DIR "org/centers/cdgc/ComputeCpp/OpenCL-Headers" CACHE STRING
    "Path to opencl headers")

set(CMAKE_MODULE_PATH "/org/centers/cdgc/ComputeCpp/ComputeCpp/cmake/Modules")

include(FindComputeCpp)

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

# Subdirectories to make
add_subdirectory(bfs)
