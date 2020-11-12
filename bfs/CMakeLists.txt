add_executable(bfs-sycl-naive bfs-sycl-naive.cpp)
add_sycl_to_target(TARGET bfs-sycl-naive SOURCES bfs-sycl-naive.cpp)
install(TARGETS bfs-sycl-naive RUNTIME DESTINATION bin)
