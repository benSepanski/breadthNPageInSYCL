# Right now does headers only
add_library(breadthnpageinsycl_syclutils INTERFACE)
add_library(breadthNPageInSYCL::syclUtils ALIAS breadthnpageinsycl_syclutils)

target_include_directories(breadthnpageinsycl_syclutils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
