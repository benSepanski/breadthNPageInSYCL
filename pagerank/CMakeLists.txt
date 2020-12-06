add_executable(pagerank-data-driven pagerank-data-driven.cpp
                              support.cpp)
add_sycl_to_target(TARGET pagerank-data-driven SOURCES pagerank-data-driven.cpp )
target_link_libraries(pagerank-data-driven breadthNPageInSYCL::syclUtils)

add_executable(pagerank-topology-driven pagerank-topology-driven.cpp
                              support.cpp)
add_sycl_to_target(TARGET pagerank-topology-driven SOURCES pagerank-topology-driven.cpp )
target_link_libraries(pagerank-topology-driven breadthNPageInSYCL::syclUtils)
