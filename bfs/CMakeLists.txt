add_executable(bfs-group-sched bfs-group-sched.cpp support.cpp)
add_sycl_to_target(TARGET bfs-group-sched SOURCES bfs-group-sched.cpp)
target_link_libraries(bfs-group-sched breadthNPageInSYCL::syclUtils)
