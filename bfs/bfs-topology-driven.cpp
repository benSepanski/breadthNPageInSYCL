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
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       node_data;
    sycl::accessor<bool, 1,
                   sycl::access::mode::write,
                   sycl::access::target::global_buffer>
                       done;
    /** Called at start of push scheduling */
    void initialize(const sycl::nd_item<1> &my_item) { }

    /** Constructor **/
    BFSOperatorInfo( SYCL_CSR_Graph &sycl_graph,
                     sycl::buffer<bool, 1> &done_buf,
                     sycl::handler &cgh )
        : node_data{ sycl_graph.node_data, cgh }
        , done{ done_buf, cgh }
    { }
    /** We must provide a copy constructor */
    BFSOperatorInfo( const BFSOperatorInfo &that )
        : node_data{ that.node_data }
        , done{ that.done }
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
        if(   opInfo.node_data[src_node] != INF 
           && opInfo.node_data[src_node] + 1 < opInfo.node_data[dst_node]) {
            opInfo.node_data[dst_node] = opInfo.node_data[src_node] + 1;
            opInfo.done[0] = false;
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
    // Initialize in-worklist to all nodes
    wl_pipe.initialize(queue);
    queue.submit([&] (sycl::handler &cgh) {
        InWorklist in_wl(wl_pipe, cgh);
        const size_t NNODES = sycl_graph.nnodes;
        cgh.parallel_for<class wl_init>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                          sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            in_wl.setSize(NNODES);
            for(index_type i = my_item.get_global_id()[0]; i < NNODES; i += NUM_WORK_ITEMS) {
                // put *i* in *i*th position
                in_wl.push(i, i);
            }
    }); });

    // Run BFS
    bool done = true;
    sycl::buffer<bool, 1> done_buf(&done, sycl::range<1>{1});
    // We won't need to rerun since we're doing topology-driven
    bool rerun = false;
    sycl::buffer<bool, 1> rerun_buf(&rerun, sycl::range<1>{1});
    while(true) {
        // Relax all edges
        queue.submit([&]( sycl::handler &cgh) {
            BFSOperatorInfo bfsInfo{ sycl_graph, done_buf, cgh };
            BFSIter current_iter(NUM_WORK_GROUPS, sycl_graph, wl_pipe, cgh, rerun_buf, bfsInfo);
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             current_iter);
        });
        // are we done?
        {
            auto done_acc = done_buf.get_access<sycl::access::mode::read_write>();
            if(done_acc[0]) {
                break;
            }
            done_acc[0] = true;
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
