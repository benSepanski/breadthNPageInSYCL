#include <chrono>
#include <iostream>
#include <limits>
#include <CL/sycl.hpp>

// From libsyclutils
//
// SYCL_CSR_Graph node_data_type index_type
#include "sycl_csr_graph.h"
// BFSPush gpu_size_t
#include "BFSPush.h"

// easier than typing cl::sycl
namespace sycl = cl::sycl;

// from support.cpp
extern index_type start_node;

// defined here, but const so need to declare as extern so support.cpp
// can use it
extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

// TODO: Make these extern consts which are determined by sycl_driver.cpp
#define NUM_THREAD_BLOCKS 8
#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32

const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
             NUM_WORK_GROUPS = NUM_THREAD_BLOCKS,
             NUM_WORK_ITEMS  = NUM_WORK_GROUPS * WORK_GROUP_SIZE,
             WARPS_PER_GROUP = WORK_GROUP_SIZE / WARP_SIZE;

// classes used to name SYCL kernels
class bfs_init;
class cumsum;
class new_worklist_size;

// BFS Sweeps
class BFSPreSweep : public BFSPush { 
    public:
    BFSPreSweep(size_t level,
                sycl::buffer<index_type, 1> &in_worklist_buf,
                sycl::buffer<size_t, 1> &in_worklist_size_buf,
                SYCL_CSR_Graph &sycl_graph,
                sycl::handler &cgh)
       : BFSPush {level, in_worklist_buf, in_worklist_size_buf, sycl_graph, cgh}
        { }
};
class BFSPostSweep : public BFSPush { 
    public:
    BFSPostSweep(size_t level,
                 sycl::buffer<index_type, 1> &in_worklist_buf,
                 sycl::buffer<size_t, 1> &in_worklist_size_buf,
                 SYCL_CSR_Graph &sycl_graph,
                 sycl::handler &cgh)
       : BFSPush {level, in_worklist_buf, in_worklist_size_buf, sycl_graph, cgh}
        { }
};

/**
 * Run BFS on the sycl_graph from start_node, storing each node's level
 * into the node_data
 */
void sycl_bfs(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    // set up worklist
    sycl::buffer<index_type, 1> worklist1_buf(sycl::range<1>{sycl_graph.nnodes}),
                                worklist2_buf(sycl::range<1>{sycl_graph.nnodes}),
                                *in_worklist_buf = &worklist1_buf,
                                *out_worklist_buf = &worklist2_buf;
    sycl::buffer<size_t, 1> in_worklist_size_buf(sycl::range<1>{1});
    sycl::buffer<gpu_size_t, 1> num_out_nodes_buf(sycl::range<1>{NUM_WORK_GROUPS});
    // initialize node levels
    queue.submit([&] (sycl::handler &cgh) {
        // get access to node level
        auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::discard_write>(cgh);
        // get access to in_worklist and in_worklist size
        auto in_worklist = in_worklist_buf->get_access<sycl::access::mode::discard_write>(cgh);
        auto in_worklist_size = in_worklist_size_buf.get_access<sycl::access::mode::discard_write>(cgh);
        // some constants
        const size_t NNODES = sycl_graph.nnodes;
        const index_type START_NODE = start_node;
        // Initialize the node data and worklists
        cgh.parallel_for<class bfs_init>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                           sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // set node levels to defaults
            for(size_t i = my_item.get_global_id()[0]; i < NNODES; i += NUM_WORK_ITEMS) {
                node_level[i] = (i == START_NODE) ? 0 : INF;
            }
            // initialize in_worklist
            if(my_item.get_global_id()[0] == 0) {
                in_worklist_size[0] = 1;
                in_worklist[0] = START_NODE;
            }
        });
    });

    size_t level = 1, in_worklist_size = 1;
    while(level < INF && in_worklist_size > 0) {
        // Run a pre-sweep of BFS
        queue.submit([&] (sycl::handler &cgh) {
            BFSPreSweep current_iter(level,
                                     *in_worklist_buf,
                                     in_worklist_size_buf,
                                     sycl_graph,
                                     cgh);
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             current_iter);
        });
        // Overwrite num_out_nodes with cumulative sum of its entries
        queue.submit([&] (sycl::handler &cgh) {
            auto array = num_out_nodes_buf.get_access<sycl::access::mode::read_write>(cgh);
            cgh.single_task<class cumsum>( [=] () {
                for(size_t i = 1; i < NUM_WORK_GROUPS; ++i) {
                    array[i] += array[i-1];
                }
            });
        });
        // Build our out-worklist
        queue.submit([&] (sycl::handler &cgh) {
            BFSPostSweep current_iter(level++,
                                      *in_worklist_buf,
                                      in_worklist_size_buf,
                                      sycl_graph,
                                      cgh);
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             current_iter);
        });
        // Get our new in-worklist size
        queue.submit([&] (sycl::handler &cgh) {
            auto in_worklist_size_acc = in_worklist_size_buf.get_access<sycl::access::mode::write>(cgh);
            auto num_out_nodes_acc = num_out_nodes_buf.get_access<sycl::access::mode::read>(cgh);
            cgh.single_task<class new_worklist_size>([=] () {
                in_worklist_size_acc[0] = num_out_nodes_acc[NUM_WORK_GROUPS-1];
            });
        });
        // swap in/out worklists
        std::swap(in_worklist_buf, out_worklist_buf);
        // get the in_worklist_size on the host 
        {
            auto in_worklist_size_acc = in_worklist_size_buf.get_access<sycl::access::mode::read>();
            in_worklist_size = in_worklist_size_acc[0];
        }
    }
    // Wait for BFS to finish and throw asynchronous errors if any
    queue.wait_and_throw();
}


int sycl_main(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    // Run sycl bfs in a try-catch block.
    try {
        sycl_bfs(sycl_graph, queue);
    } catch (cl::sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
        if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
        }
        std::exit(1);
    }

    return 0;
}

void BFSPreSweep::applyToOutEdge(sycl::nd_item<1> my_item, index_type edge_index) {
    index_type dst_node = edge_dst[edge_index];
    if(node_level[dst_node] == INF) {
        node_level[dst_node] = LEVEL;
    }
}

void BFSPostSweep::applyToOutEdge(sycl::nd_item<1> my_item, index_type edge_index) {
    index_type dst_node = edge_dst[edge_index];
    if(node_level[dst_node] == INF) {
        node_level[dst_node] = LEVEL;
    }
}
