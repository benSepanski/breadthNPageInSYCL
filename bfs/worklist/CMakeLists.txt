add_executable(bfs-worklist bfs-worklist.cpp ../support.cpp BFSPush.h BFSPush.cpp)
add_sycl_to_target(TARGET bfs-worklist SOURCES bfs-worklist.cpp BFSPush.h BFSPush.cpp)
target_link_libraries(bfs-worklist breadthNPageInSYCL::syclUtils)
