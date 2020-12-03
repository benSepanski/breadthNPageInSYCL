add_executable(pagerank-naive pagerank-naive.cpp
                              support.cpp)
add_sycl_to_target(TARGET pagerank-naive SOURCES pagerank-naive.cpp )
target_link_libraries(pagerank-naive breadthNPageInSYCL::syclUtils)
