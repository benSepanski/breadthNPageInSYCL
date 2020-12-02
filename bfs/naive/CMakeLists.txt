add_executable(bfs-naive bfs-naive.cpp ../support.cpp)
add_sycl_to_target(TARGET bfs-naive SOURCES bfs-naive.cpp)
target_link_libraries(bfs-naive breadthNPageInSYCL::syclUtils)
