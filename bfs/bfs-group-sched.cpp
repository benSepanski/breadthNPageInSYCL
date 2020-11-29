#include <chrono>
#include <iostream>
#include <limits>
#include <CL/cl.h>
#include <CL/sycl.hpp>

// From libsyclutils
//
// SYCL_CSR_Graph node_data_type index_type
#include "sycl_csr_graph.h"

// easier than typing cl::sycl
namespace sycl = cl::sycl;

// from support.cpp
extern index_type start_node;

// NVIDIA target can't do 64-bit atomic adds, even though it says it can
// sycl_driver makes sure this is big enough
typedef uint32_t gpu_size_t;

// defined here, but const so need to declare as extern so support.cpp
// can use it
extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

// TODO: Make these extern consts which are determined by sycl_driver.cpp
#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32

// classes used to name SYCL kernels
class bfs_init;
class bfs_iter;


/**
 * Run BFS on the sycl_graph from start_node, storing each node's level
 * into the node_data
 */
void sycl_bfs(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    /// Put some constants on a buffer and initialize to avoid host-device loops ////////////////////////////
    //
    const index_type NNODES = sycl_graph.nnodes,
                     START_NODE = start_node;
    sycl::buffer<index_type, 1> NNODES_buf(&NNODES, sycl::range<1>{1}),
                                START_NODE_buf(&START_NODE, sycl::range<1>{1}),
                                INF_buf(&INF, sycl::range<1>{1});
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /// Initialize the node data ////////////////////////////////////////////////////////////////////////////
    //
    // initialize node levels
    queue.submit([&] (sycl::handler &cgh) {
        // get access to node level
        auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::discard_write>(cgh);
        // get INF on the device
        auto INF_acc = INF_buf.get_access<sycl::access::mode::read,
                                          sycl::access::target::constant_buffer>(cgh);
        // need access to start node
        auto START_NODE_acc = START_NODE_buf.get_access<sycl::access::mode::read,
                                                        sycl::access::target::constant_buffer>(cgh);
        auto NNODES_acc = NNODES_buf.get_access<sycl::access::mode::read,
                                                sycl::access::target::constant_buffer>(cgh);
        // Initialize the node data
        const size_t BFS_INIT_WORK_GROUP_SIZE = std::min((size_t) THREAD_BLOCK_SIZE, (size_t) NNODES);
        // round up NNODES to work-group size
        const size_t NUM_INIT_WORK_GROUPS = NNODES
                                           + (BFS_INIT_WORK_GROUP_SIZE - NNODES % BFS_INIT_WORK_GROUP_SIZE)
                                           % BFS_INIT_WORK_GROUP_SIZE;
        cgh.parallel_for<class bfs_init>(sycl::nd_range<1>{sycl::range<1>{NUM_INIT_WORK_GROUPS},
                                                           sycl::range<1>{BFS_INIT_WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // make sure global id is valid
            if(my_item.get_global_id()[0] >= NNODES_acc[0]) {
                return;
            }
            // initialize everything
            node_level[my_item.get_global_id()] = (my_item.get_global_id()[0] == START_NODE_acc[0]) ? 0 : INF_acc[0];
        });
    });
    // Because we used discard writes, for some reason
    // sycl does not pick up on the data and dependency and
    // wait for the initialization to finish before starting the iters.
    // So, we must force it to wait explicitly
    queue.wait();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Get some work-distribution constants onto device to avoid copies
    const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
    sycl::buffer<const size_t, 1> WORK_GROUP_SIZE_buf(&WORK_GROUP_SIZE, sycl::range<1>{1});
    // assume warp size evenly divides work-group size
    assert( WORK_GROUP_SIZE % WARP_SIZE == 0 );
    const size_t WARPS_PER_GROUP = WORK_GROUP_SIZE / WARP_SIZE;
    // min degree to work on node as group
    const size_t MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE;
    //const size_t MIN_GROUP_SCHED_DEGREE = 1;
    sycl::buffer<const size_t, 1> MIN_GROUP_SCHED_DEGREE_buf(&MIN_GROUP_SCHED_DEGREE, sycl::range<1>{1});
    // min degree to work on node as warp
    const size_t MIN_WARP_SCHED_DEGREE = 1;
    sycl::buffer<const size_t, 1> MIN_WARP_SCHED_DEGREE_buf(&MIN_WARP_SCHED_DEGREE, sycl::range<1>{1});
    // We need to know when we've hit the last iteration
    bool new_nodes = false, new_nodes_local_copy = false;
    sycl::buffer<bool, 1> new_nodes_buf(&new_nodes, sycl::range<1>{1});
    /// Now invoke BFS routine at each level ////////////////////////////////////////////////////////////////
    size_t level = 1, level_host_copy = 1;
    sycl::buffer<size_t, 1> LEVEL_buf(&level, sycl::range<1>{1});
    do {
        /// Run an iteration of BFS ///////////////////////////////////////////////////////////////
        queue.submit([&] (sycl::handler &cgh) {
            sycl::stream sycl_stream(1024, 1024, cgh);
            // constants for work distribution
            const size_t NUM_WORK_GROUPS = (NNODES + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

            // get constant access to some work-distribution constantss size on device
            auto WORK_GROUP_SIZE_acc = WORK_GROUP_SIZE_buf.get_access<sycl::access::mode::read,
                                                                      sycl::access::target::constant_buffer>(cgh);
            auto MIN_GROUP_SCHED_DEGREE_acc = MIN_GROUP_SCHED_DEGREE_buf.get_access<sycl::access::mode::read,
                                                                                    sycl::access::target::constant_buffer>(cgh);
            auto MIN_WARP_SCHED_DEGREE_acc = MIN_WARP_SCHED_DEGREE_buf.get_access<sycl::access::mode::read,
                                                                                  sycl::access::target::constant_buffer>(cgh);
            // Get read-access to the CSR graph
            auto row_start = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);
            auto edge_dst  = sycl_graph.edge_dst.get_access<sycl::access::mode::read>(cgh);
            // Get read-write access to the node's level
            auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::read_write>(cgh);
            // Get constant access to the number of nodes and the INF constant
            auto NNODES_acc = NNODES_buf.get_access<sycl::access::mode::read,
                                                    sycl::access::target::constant_buffer>(cgh);
            auto INF_acc = INF_buf.get_access<sycl::access::mode::read,
                                              sycl::access::target::constant_buffer>(cgh);
            // We need to know if we have to keep going or not
            auto new_nodes_acc = new_nodes_buf.get_access<sycl::access::mode::write>(cgh);
            // we'll need the current level
            auto LEVEL = LEVEL_buf.get_access<sycl::access::mode::read>(cgh);

            // group-local memory:
            cl::sycl::accessor<index_type, 1,
                               sycl::access::mode::read_write,
                               sycl::access::target::local> 
                                   // local memory for nodes to store their first/last edges
                                   group_first_edges(sycl::range<1>{WORK_GROUP_SIZE}, cgh),
                                   group_last_edges(sycl::range<1>{WORK_GROUP_SIZE}, cgh),
                                   // local memory for a work-node during group-scheduling
                                   group_work_node(sycl::range<1>{1}, cgh),
                                   // local memory for work-nodes during warp-scheduling
                                   warp_work_node(sycl::range<1>{WARPS_PER_GROUP}, cgh),
                                   // communication between warps
                                   warp_still_has_work(sycl::range<1>{1}, cgh);

            /// Now submit our bfs job //////////////////////////////////////////////////
            const size_t GLOBAL_SIZE = NUM_WORK_GROUPS * WORK_GROUP_SIZE;
            cgh.parallel_for<class bfs_iter>(sycl::nd_range<1>{sycl::range<1>{GLOBAL_SIZE},
                                                               sycl::range<1>{WORK_GROUP_SIZE}},
            [=] (sycl::nd_item<1> my_item) {
                // figure out what work I need to do (if any)
                index_type my_work_left = 0;
                if(my_item.get_global_id()[0] < NNODES_acc[0]
                   && node_level[my_item.get_global_id()] == (LEVEL[0]-1)) {
                    // get my src node, first edge, and last edge in private memory
                    index_type my_node = my_item.get_global_id()[0],
                               first_edge = row_start[my_node],
                                last_edge = row_start[my_node+1];
                    // put first/last edge into local memory
                    group_first_edges[my_item.get_local_id()] = first_edge;
                    group_last_edges[my_item.get_local_id()] = last_edge;
                    // store the work I have left to do in private memory
                    my_work_left = last_edge - first_edge;
                }
                // Initialize work_node to its invalid value
                group_work_node[0] = WORK_GROUP_SIZE_acc[0];
                // Now wait for everyone else in my group to figure their work too
                my_item.barrier(sycl::access::fence_space::local_space);

                /// group-scheduling //////////////////////////////////////////
                //(work as group til no one wants group control)
                while(true) {
                    // If I have enough work to do that I want to control the
                    // whole group, bid for control!
                    if( my_work_left >= MIN_GROUP_SCHED_DEGREE_acc[0] ) {
                        group_work_node[0] = my_item.get_local_id()[0];
                    }
                    // Wait for everyone's control bids to finalize
                    my_item.barrier(sycl::access::fence_space::local_space);
                    // If no-one competed for control of the group, we're done!
                    if(group_work_node[0] == WORK_GROUP_SIZE_acc[0]) {
                        break;
                    }
                    // Otherwise, copy the work node into private memory
                    // and clear the group-work-node for next time
                    index_type work_node = group_work_node[0];
                    my_item.barrier(sycl::access::fence_space::local_space);
                    if( work_node == my_item.get_local_id()[0] ) {
                        group_work_node[0] = WORK_GROUP_SIZE_acc[0];
                        my_work_left = 0;
                    }
                    my_item.barrier(sycl::access::fence_space::local_space);

                    // Now work on the work_node's out-edges in batches of
                    // size WORK_GROUP_SIZE
                    size_t current_edge = group_first_edges[work_node] + my_item.get_local_id()[0],
                              last_edge = group_last_edges[work_node];
                    while( current_edge < last_edge ) {
                        index_type dst_node = edge_dst[current_edge];
                        if( node_level[dst_node] > LEVEL[0] ) {
                            node_level[dst_node] = LEVEL[0];
                            new_nodes_acc[0] = true;
                        }
                        current_edge += WORK_GROUP_SIZE_acc[0];
                    }
                }
                ///////////////////////////////////////////////////////////////
                //
                // set up for warp scheduling
                warp_still_has_work[0] = false;
                size_t warp_id = my_item.get_local_id()[0] / WARP_SIZE,
                       my_warp_local_id = my_item.get_local_id()[0] % WARP_SIZE;
                warp_work_node[warp_id] = WORK_GROUP_SIZE_acc[0];
                // wait for group-scheduling to finish
                my_item.barrier();

                /// warp-scheduling //////////////////////////////////////////
                //(work as warps til no one wants warp control)
                while(true) {
                    // If I have enough work to do that I want to control the
                    // warp, bid for control!
                    if( my_work_left >= MIN_WARP_SCHED_DEGREE_acc[0] ) {
                        warp_work_node[warp_id] = my_item.get_local_id()[0];
                        warp_still_has_work[0] = true;
                    }
                    // Wait for everyone's control bids to finalize
                    my_item.barrier(sycl::access::fence_space::local_space);
                    // If no-one competed for control of the group, we're done!
                    if(!warp_still_has_work[0]) {
                        break;
                    }
                    // Otherwise, copy the work node into private memory
                    // and set warp_still_has_work to false for next time
                    index_type work_node = warp_work_node[warp_id];
                    my_item.barrier(sycl::access::fence_space::local_space);
                    if( work_node == my_item.get_local_id()[0] ) {
                        warp_work_node[warp_id] = WORK_GROUP_SIZE_acc[0];
                        warp_still_has_work[0] = false;
                        my_work_left = 0;
                    }
                    my_item.barrier(sycl::access::fence_space::local_space);
                    // if my warp has no work to-do, just keep waiting in
                    // this while-loop (so that other warps don't deadlock
                    //                  on the barriers)
                    if(work_node >= WORK_GROUP_SIZE_acc[0]) {
                        continue;
                    }

                    // Now work on the work_node's out-edges in batches of
                    // size WARP_SIZE 
                    size_t current_edge = group_first_edges[work_node] + my_warp_local_id,
                              last_edge = group_last_edges[work_node];
                    while( current_edge < last_edge ) {
                        index_type dst_node = edge_dst[current_edge];
                        if( node_level[dst_node] > LEVEL[0] ) {
                            node_level[dst_node] = LEVEL[0];
                            new_nodes_acc[0] = true;
                        }
                        current_edge += WARP_SIZE;
                    }
                }
                ///////////////////////////////////////////////////////////////
            });
            /////////////////////////////////////////////////////////////////////////////
        });
        ///////////////////////////////////////////////////////////////////////////////////////////

        /// Get ready for next iteration //////////////////////////////////////////////////////////
        //
        // Wait for iteration to finish and throw asynchronous errors if any
        queue.wait_and_throw();
        // Begin Update Level SYCL scope (buffers only block to write-out on destruction,
        //                          so without this scope the level may not increment, etc.)
        {
            // NOTE: one would think that asking for write-access is enough to block,
            //       but you HAVE TO ASK FOR READ ACCESS HERE in order to block
            //       the next bfs iteration from running before the level is incremented
            // Increment level (avoiding a copy device->host)
            auto level_acc = LEVEL_buf.get_access<sycl::access::mode::read_write>();
            level_acc[0] = ++level_host_copy;
            auto new_nodes_acc = new_nodes_buf.get_access<sycl::access::mode::read_write>();
            new_nodes_local_copy = new_nodes_acc[0];
            new_nodes_acc[0] = false;
            //std::cerr << level_host_copy << std::endl;
        } // End Update Level SYCL scope
        ///////////////////////////////////////////////////////////////////////////////////////////
    } while(level_host_copy < INF && new_nodes_local_copy);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
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
