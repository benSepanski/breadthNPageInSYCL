add_executable(bfs-depth bfs-depth.cpp ../support.cpp)
add_sycl_to_target(TARGET bfs-depth SOURCES bfs-depth.cpp)
target_link_libraries(bfs-depth breadthNPageInSYCL::syclUtils)
