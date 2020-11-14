add_executable(bfs-sycl-naive bfs-sycl-naive.cpp)
add_sycl_to_target(TARGET bfs-sycl-naive SOURCES bfs-sycl-naive.cpp)
# need private for sycl linking
target_link_libraries(bfs-sycl-naive PRIVATE Galois::shmem)
