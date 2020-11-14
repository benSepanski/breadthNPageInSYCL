add_executable(pagerank-sycl-naive pagerank-sycl-naive.cpp)
target_link_libraries(pagerank-sycl-naive Galois::shmem)
add_sycl_to_target(TARGET pagerank-sycl-naive SOURCES pagerank-sycl-naive.cpp)
