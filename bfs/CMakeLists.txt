add_executable(bfs-data-driven bfs-data-driven.cpp support.cpp )
add_sycl_to_target(TARGET bfs-data-driven SOURCES bfs-data-driven.cpp)
target_link_libraries(bfs-data-driven breadthNPageInSYCL::syclUtils)

add_executable(bfs-topology-driven bfs-topology-driven.cpp support.cpp )
add_sycl_to_target(TARGET bfs-topology-driven SOURCES bfs-topology-driven.cpp)
target_link_libraries(bfs-topology-driven breadthNPageInSYCL::syclUtils)
