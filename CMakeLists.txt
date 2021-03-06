# Built following the codeplay guide:
# https://developer.codeplay.com/products/computecpp/ce/guides/integration-guide

project(breadthNPageInSYCL)

cmake_minimum_required(VERSION 3.4.3)
set(CMAKE_CXX_STANDARD 17)

####### Submodule setup from https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
#######
#######  Convenient for Galois submodule
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Galois/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()


####### ComputeCPP setup
set(CMAKE_MODULE_PATH "/org/centers/cdgc/ComputeCpp/ComputeCpp/cmake/Modules")
include(FindComputeCpp)

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

### Build Galois since we need LonestarGPU apps
add_subdirectory(extern/Galois EXCLUDE_FROM_ALL)

### Use write opencl version
add_compile_definitions(CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})

### Other subdirectories to make
add_subdirectory(libsyclutils)
add_subdirectory(bfs)
add_subdirectory(pagerank)
