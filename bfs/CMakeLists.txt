add_executable(bfs-naive bfs-naive.cpp support.cpp)
add_sycl_to_target(TARGET bfs-naive SOURCES bfs-naive.cpp)
target_link_libraries(bfs-naive breadthNPageInSYCL::syclUtils)

add_executable(bfs-batched-submit bfs-batched-submit.cpp support.cpp)
add_sycl_to_target(TARGET bfs-batched-submit SOURCES bfs-batched-submit.cpp)
target_link_libraries(bfs-batched-submit breadthNPageInSYCL::syclUtils)

add_executable(bfs-depth bfs-depth.cpp support.cpp)
add_sycl_to_target(TARGET bfs-depth SOURCES bfs-depth.cpp)
target_link_libraries(bfs-depth breadthNPageInSYCL::syclUtils)
