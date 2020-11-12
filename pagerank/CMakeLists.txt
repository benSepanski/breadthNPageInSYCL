add_executable(pagerank-sycl-naive pagerank-sycl-naive.cpp)
add_sycl_to_target(TARGET pagerank-sycl-naive SOURCES pagerank-sycl-naive.cpp)
