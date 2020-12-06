#include <iostream>

// From libsyclutils
//
// THREAD_BLOCK_SIZE WARP_SIZE
#include "kernel_sizing.h"
// SYCL_CSR_Graph node_data_type index_type
#include "sycl_csr_graph.h"
// Pipe
#include "pipe.h"
// PushScheduler INF
#include "push_scheduler.h"

// easier than typing cl::sycl
namespace sycl = cl::sycl;

// class names for SYCL kernels
class bfs_init;
class wl_init;

// from support.cpp
extern index_type start_node;

extern size_t num_work_groups;

struct BFSOperatorInfo {
    node_data_type level;
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       node_data;
    /** Called at start of push scheduling */
    void initialize(const sycl::nd_item<1> &my_item) { }

    /** Constructor **/
    BFSOperatorInfo( SYCL_CSR_Graph &sycl_graph, sycl::handler &cgh, node_data_type level ) 
        : node_data{ sycl_graph.node_data, cgh }
        , level{ level }
    { }
    /** We must provide a copy constructor */
    BFSOperatorInfo( const BFSOperatorInfo &that )
        : node_data{ that.node_data }
        , level{ that.level }
    { }
};


// Define our BFS push operator
class BFSIter : public PushScheduler<BFSIter, BFSOperatorInfo> {
    public:
    BFSIter(gpu_size_t num_work_groups,
            SYCL_CSR_Graph &sycl_graph, Pipe &pipe, sycl::handler &cgh,
            sycl::buffer<bool, 1> &out_worklist_needs_compression,
            BFSOperatorInfo &opInfo)
        : PushScheduler{num_work_groups, sycl_graph, pipe, cgh, out_worklist_needs_compression, opInfo}
        { }

    void applyPushOperator(const sycl::nd_item<1>&,
                           index_type src_node,
                           index_type edge_index)
    {
        // invalid edge case
        if(edge_index >= NEDGES) return;
        // valid edge case
        index_type dst_node = edge_dst[edge_index];
        if(opInfo.node_data[dst_node] == INF) {
            bool push_success = out_wl.push(dst_node);
            if(push_success) {
                opInfo.node_data[dst_node] = opInfo.level;
            }
            else {
                out_worklist_full[0] = true;
            }
        }
    }
};


/**
 * Run BFS on the sycl_graph from start_node, storing each node's level
 * into the node_data
 */
void sycl_bfs(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                 NUM_WORK_GROUPS = num_work_groups,
                 NUM_WORK_ITEMS  = NUM_WORK_GROUPS * WORK_GROUP_SIZE,
                 WARPS_PER_GROUP = WORK_GROUP_SIZE / WARP_SIZE;
    // set up worklists
    Pipe wl_pipe{(gpu_size_t) sycl_graph.nnodes,
                 (gpu_size_t) sycl_graph.nnodes,
                 (gpu_size_t) NUM_WORK_GROUPS};

    // initialize node levels
    queue.submit([&] (sycl::handler &cgh) {
        // get access to node level
        auto node_data = sycl_graph.node_data.get_access<sycl::access::mode::discard_write>(cgh);
        // some constants
        const size_t NNODES = sycl_graph.nnodes;
        const index_type START_NODE = start_node;
        // Initialize the node data and worklists
        cgh.parallel_for<class bfs_init>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                           sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // set node levels to defaults
            for(size_t i = my_item.get_global_id()[0]; i < NNODES; i += NUM_WORK_ITEMS) {
                node_data[i] = (i == START_NODE) ? 0 : INF;
            }
        });
    });
    // Initialize in-worklist
    wl_pipe.initialize(queue);
    queue.submit([&] (sycl::handler &cgh) {
        InWorklist in_wl(wl_pipe, cgh);
        const index_type START_NODE = start_node;
        cgh.single_task<class wl_init>( [=]() {
            in_wl.setSize(1);
            in_wl.push(0, START_NODE);
        });
    });

    // Run BFS
    size_t level = 1;
    bool rerun_level = false;
    sycl::buffer<bool, 1> rerun_level_buf(&rerun_level, sycl::range<1>{1});
    gpu_size_t in_wl_size = 1;
    while(in_wl_size > 0) {
        queue.submit([&]( sycl::handler &cgh) {
            BFSOperatorInfo bfsInfo{ sycl_graph, cgh, level };
            BFSIter current_iter(NUM_WORK_GROUPS, sycl_graph, wl_pipe, cgh, rerun_level_buf, bfsInfo);
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             current_iter);
        });

        wl_pipe.compress(queue);
        {
            auto rerun_level_acc = rerun_level_buf.get_access<sycl::access::mode::read_write>();
            if(!rerun_level_acc[0]) {
                level++;
                wl_pipe.swapSlots(queue);
                auto in_wl_size_acc = wl_pipe.get_in_worklist_size_buf().get_access<sycl::access::mode::read>();
                in_wl_size = in_wl_size_acc[0];
            }
            rerun_level_acc[0] = false;
        }
    }
    // Wait for BFS to finish and throw asynchronous errors if any
    queue.wait_and_throw();
}


int sycl_main(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    std::cout << "NUM WORK GROUPS: " << num_work_groups << "\n";
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
