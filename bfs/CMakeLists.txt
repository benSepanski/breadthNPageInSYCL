add_executable(bfs-sycl-naive bfs-sycl-naive.cpp support.cpp)
add_sycl_to_target(TARGET bfs-sycl-naive SOURCES bfs-sycl-naive.cpp)
target_link_libraries(bfs-sycl-naive breadthNPageInSYCL::syclUtils)

add_executable(bfs-sycl-group-sched bfs-sycl-group-sched.cpp support.cpp)
add_sycl_to_target(TARGET bfs-sycl-group-sched SOURCES bfs-sycl-group-sched.cpp)
target_link_libraries(bfs-sycl-group-sched breadthNPageInSYCL::syclUtils)
