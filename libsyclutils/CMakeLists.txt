# Right now does headers only
add_library(breadthnpageinsycl_syclutils)
add_library(breadthNPageInSYCL::syclUtils ALIAS breadthnpageinsycl_syclutils)

target_sources( breadthnpageinsycl_syclutils PRIVATE
    include/host_csr_graph.h
    include/nvidia_selector.h
    src/host_csr_graph.cpp
    src/nvidia_selector.cpp
    src/sycl_driver.cpp
)

target_include_directories(breadthnpageinsycl_syclutils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
