#include <climits>
#include <iostream>
#include <CL/sycl.hpp>

// SYCL_CSR_Graph node_data_type index_type
#include "sycl_csr_graph.h"
// SYCLPipe
#include "sycl-pipe.h"
// SYCLOutWorklist
#include "sycl-out-worklist.h"

#define THREAD_BLOCK_SIZE 256

extern const float ALPHA = 0.85;
extern const float EPSILON = 0.000001;
extern int MAX_ITERATIONS ;

namespace sycl = cl::sycl;

class TEST;
class TEST2;
class TEST3;

// probability of each node as computed by pagerank
float *P_CURR;

// declaration of pagerank function, which wil be called by main
void sycl_pagerank(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue);

int sycl_main(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    try {
        sycl_pagerank(sycl_graph, queue);
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
        if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
        }
        std::exit(1);
    }
                
   return 0;
}


void sycl_pagerank(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    const gpu_size_t NUM_WORK_GROUPS = 6;
    SYCLPipe wl_pipe{(gpu_size_t) sycl_graph.nedges, NUM_WORK_GROUPS};
    P_CURR = new float[sycl_graph.nnodes];

    wl_pipe.initialize(queue);

    queue.submit([&] (sycl::handler &cgh) {
        SYCLOutWorklist out_wl(wl_pipe, cgh);

        const gpu_size_t NNODES = (gpu_size_t) sycl_graph.nnodes;
        const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
        const gpu_size_t NUM_WORK_ITEMS = NUM_WORK_GROUPS * WORK_GROUP_SIZE;
        cgh.parallel_for<class TEST>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                       sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // pull global info about out-worklist into local memory
            if(my_item.get_local_id()[0] == 0) {
                out_wl.initializeLocalMemory(my_item);
            }
            my_item.barrier(sycl::access::fence_space::local_space);

            if(my_item.get_local_id()[0] == 0) {
                out_wl.push(my_item, my_item.get_group(0));
            }
            // publish any local modifications to the out-worklist
            // back into global memory
            my_item.barrier(sycl::access::fence_space::local_space);
            if(my_item.get_local_id()[0] == 0) {
                out_wl.publishLocalMemory(my_item);
            }
        });
    });
    wl_pipe.compress(queue);
    queue.submit([&] (sycl::handler &cgh) {
        SYCLOutWorklist out_wl(wl_pipe, cgh);

        const gpu_size_t NNODES = (gpu_size_t) sycl_graph.nnodes;
        const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
        const gpu_size_t NUM_WORK_ITEMS = NUM_WORK_GROUPS * WORK_GROUP_SIZE;
        cgh.parallel_for<class TEST3>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                       sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // pull global info about out-worklist into local memory
            if(my_item.get_local_id()[0] == 0) {
                out_wl.initializeLocalMemory(my_item);
            }
            my_item.barrier(sycl::access::fence_space::local_space);

            out_wl.push(my_item, my_item.get_group(0));
            // publish any local modifications to the out-worklist
            // back into global memory
            my_item.barrier(sycl::access::fence_space::local_space);
            if(my_item.get_local_id()[0] == 0) {
                out_wl.publishLocalMemory(my_item);
            }
        });
    });
    wl_pipe.compress(queue);
    queue.submit([&] (sycl::handler &cgh) {
        sycl::stream sycl_stream(1025, 256, cgh);
        SYCLOutWorklist out_wl(wl_pipe, cgh);
        cgh.single_task<class TEST2>([=]() {
            out_wl.print(sycl_stream);
        });
    });
    queue.wait_and_throw();
    wl_pipe.swapSlots(queue);
}
