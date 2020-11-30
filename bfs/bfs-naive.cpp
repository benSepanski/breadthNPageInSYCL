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
// One iteration of BFS
class BFS_iter{
    public:
    static const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
    private: 
    // work distribution constants
    const size_t NNODES,
                 NEDGES,
                 LEVEL,
                 WARPS_PER_GROUP = (WORK_GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE,
                 MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE,
                 MIN_WARP_SCHED_DEGREE = WARP_SIZE,
                 FINE_GRAINED_EDGE_CAP = WORK_GROUP_SIZE;
    // global memory:
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // read-access to the CSR graph
                       row_start,
                       edge_dst;
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // read-write access to the BFS level of each node
                       node_level;
    sycl::accessor<bool, 1,
                   sycl::access::mode::write,
                   sycl::access::target::global_buffer>
                       // set to true iff we made updates
                       made_updates;
    // group-local memory:
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local> 
                       // local memory for nodes to store their first/last edges
                       group_first_edges,
                       group_last_edges,
                       // local memory for a work-node during group-scheduling
                       group_work_node,
                       // local memory for work-nodes during warp-scheduling
                       warp_work_node,
                       // fine-grained scheduling for edges
                       fine_grained_edges;
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local> 
                       // communication between warps
                       warp_still_has_work;
    sycl::accessor<gpu_size_t, 1,
                   sycl::access::mode::atomic,
                   sycl::access::target::local>
                       // fine-grained scheduling queue size
                       num_fine_grained_edges;

    void group_scheduling(const sycl::nd_item<1> &my_item,
                          index_type &my_work_left,
                          index_type my_first_edge, 
                          index_type my_last_edge);

    void warp_scheduling(const sycl::nd_item<1> &my_item,
                         index_type &my_work_left,
                         index_type my_first_edge, 
                         index_type my_last_edge);

    void fine_grained_scheduling(const sycl::nd_item<1> &my_item,
                                 index_type &my_work_left,
                                 index_type my_first_edge, 
                                 index_type my_last_edge);

    public:
    BFS_iter(size_t level,
             sycl::buffer<bool, 1> &made_updates_buf,
             SYCL_CSR_Graph &sycl_graph,
             sycl::handler &cgh)
        // initialize some constants
        : NNODES{ sycl_graph.nnodes }
        , NEDGES{ sycl_graph.nedges }
        , LEVEL{ level }
        // to record if we made updates
        , made_updates{ made_updates_buf, cgh}
        // group-local memory
        , group_first_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
        , group_last_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
        , group_work_node{ sycl::range<1>{1}, cgh }
        , warp_work_node{ sycl::range<1>{WARPS_PER_GROUP}, cgh }
        , warp_still_has_work{ sycl::range<1>{1}, cgh }
        , fine_grained_edges{ sycl::range<1>{FINE_GRAINED_EDGE_CAP}, cgh }
        , num_fine_grained_edges{ sycl::range<1>{1}, cgh }
        // graph in global memory
        , row_start{ sycl_graph.row_start, cgh }
        , edge_dst{  sycl_graph.edge_dst,  cgh }
        , node_level{sycl_graph.node_data, cgh }
        { }

    void operator()(sycl::nd_item<1>);
};

/**
 * Run BFS on the sycl_graph from start_node, storing each node's level
 * into the node_data
 */
void sycl_bfs(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    // initialize node levels
    queue.submit([&] (sycl::handler &cgh) {
        // get access to node level
        auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::discard_write>(cgh);
        const size_t NNODES = sycl_graph.nnodes;
        const index_type START_NODE = start_node;
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
            sycl::id<1> my_global_id = my_item.get_global_id();
            if(my_global_id[0] >= NNODES) {
                return;
            }
            // initialize everything
            node_level[my_global_id] = (my_global_id[0] == START_NODE) ? 0 : INF;
        });
    });
    // Because we used discard writes, for some reason
    // sycl does not pick up on the data and dependency and
    // wait for the initialization to finish before starting the iters.
    // So, we must force it to wait explicitly
    queue.wait();
    const size_t WORK_GROUP_SIZE = BFS_iter::WORK_GROUP_SIZE,
                 NUM_WORK_GROUPS = (sycl_graph.nnodes + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE,
                 NUM_WORK_ITEMS = NUM_WORK_GROUPS * WORK_GROUP_SIZE;
    bool made_updates = false,
         made_updates_local_copy = made_updates;
    size_t level = 1;
    sycl::buffer<bool, 1> made_updates_buf(&made_updates, sycl::range<1>{1});
    do {
        // Run an iteration of BFS
        queue.submit([&] (sycl::handler &cgh) {
            BFS_iter current_iter(level++, made_updates_buf, sycl_graph, cgh);
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             current_iter);
        });
        // Wait for iteration to finish and throw asynchronous errors if any
        queue.wait_and_throw();
        // Begin Update Level SYCL scope (buffers only block to write-out on destruction,
        //                          so without this scope the level may not increment, etc.)
        {
            // NOTE: one would think that asking for write-access is enough to block,
            //       but you HAVE TO ASK FOR READ ACCESS HERE in order to block
            //       the next bfs iteration from running before the level is incremented
            // Increment level (avoiding a copy device->host)
            auto made_updates_acc = made_updates_buf.get_access<sycl::access::mode::read_write>();
            made_updates_local_copy = made_updates_acc[0];
            made_updates_acc[0] = false;
        } // End Update Level SYCL scope
    } while(level < INF && made_updates_local_copy);
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

/**
 * Run group-scheduling on an iteration of BFS
 *
 * Should only be called internally by BFS_iter.
 * Works on nodes with degree >= MIN_GROUP_SCHED_DEGREE
 * as a group.
 * Sets my_work_left to 0 if my node got worked on.
 *
 * Barriers:
 *   - global and local at start and end.
 *
 * my_item: sycl object representing my item
 * my_work_left: the amount of work my item still wants done.
 *               May be modified.
 * my_first_edge: the edge index of the first out-edge from the node
 *                I want worked on, if any
 * my_first_edge: the edge index of the last out-edge from the node
 *                I want worked on, if any
 */
void BFS_iter::group_scheduling(const sycl::nd_item<1> &my_item,
                                index_type &my_work_left,
                                index_type my_first_edge, 
                                index_type my_last_edge)
{
    my_item.barrier(sycl::access::fence_space::local_space);
    // Initialize work_node to its invalid value
    group_work_node[0] = WORK_GROUP_SIZE;
    // Now wait for everyone 
    my_item.barrier(sycl::access::fence_space::global_and_local);
    //(work as group til no one wants group control)
    while(true) {
        // If I have enough work to do that I want to control the
        // whole group, bid for control!
        if( my_work_left >= MIN_GROUP_SCHED_DEGREE ) {
            group_work_node[0] = my_item.get_local_id()[0];
        }
        // Wait for everyone's control bids to finalize
        // (and for any node-level updates to stick)
        my_item.barrier(sycl::access::fence_space::global_and_local);
        // If no-one competed for control of the group, we're done!
        if(group_work_node[0] == WORK_GROUP_SIZE) {
            break;
        }
        // Otherwise, copy the work node into private memory
        // and clear the group-work-node for next time
        index_type work_node = group_work_node[0];
        my_item.barrier(sycl::access::fence_space::local_space);
        if( work_node == my_item.get_local_id()[0] ) {
            group_work_node[0] = WORK_GROUP_SIZE;
            my_work_left = 0;
        }
        my_item.barrier(sycl::access::fence_space::local_space);

        // Now work on the work_node's out-edges in batches of
        // size WORK_GROUP_SIZE
        size_t current_edge = group_first_edges[work_node] + my_item.get_local_id()[0],
                  last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            index_type dst_node = edge_dst[current_edge];
            if( node_level[dst_node] > LEVEL ) {
                node_level[dst_node] = LEVEL;
                made_updates[0] = true;
            }
            current_edge += WORK_GROUP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run warp-scheduling on an iteration of BFS
 *
 * Should only be called internally by BFS_iter.
 * Works on nodes with 
 * MIN_WARP_SCHED_DEGREE <= degree < MIN_GROUP_SCHED_DEGREE
 * as warps.
 * Sets my_work_left to 0 if my node got worked on.
 *
 * Barriers:
 *   - global and local at start and end.
 *
 * my_item: sycl object representing my item
 * my_work_left: the amount of work my item still wants done.
 *               May be modified.
 * my_first_edge: the edge index of the first out-edge from the node
 *                I want worked on, if any
 * my_first_edge: the edge index of the last out-edge from the node
 *                I want worked on, if any
 */
void BFS_iter::warp_scheduling(const sycl::nd_item<1> &my_item,
                               index_type &my_work_left,
                               index_type my_first_edge, 
                               index_type my_last_edge)
{
    my_item.barrier(sycl::access::fence_space::local_space);
    // set up for warp scheduling
    warp_still_has_work[0] = false;
    size_t warp_id = my_item.get_local_id()[0] / WARP_SIZE,
           my_warp_local_id = my_item.get_local_id()[0] % WARP_SIZE;
    warp_work_node[warp_id] = WORK_GROUP_SIZE;
    // Wait for memory consistency
    my_item.barrier(sycl::access::fence_space::global_and_local);

    //(work as warps til no one wants warp control)
    while(true) {
        // If I have enough work to do that I want to control the
        // warp, bid for control!
        if( MIN_WARP_SCHED_DEGREE <= my_work_left && my_work_left < MIN_GROUP_SCHED_DEGREE ) {
            warp_work_node[warp_id] = my_item.get_local_id()[0];
            warp_still_has_work[0] = true;
        }
        // Wait for everyone's control bids to finalize
        // (and for any level updates to stick)
        my_item.barrier(sycl::access::fence_space::global_and_local);
        // If no-one competed for control of the group, we're done!
        if(!warp_still_has_work[0]) {
            break;
        }
        // Otherwise, copy the work node into private memory
        // and set warp_still_has_work to false for next time
        index_type work_node = warp_work_node[warp_id];
        my_item.barrier(sycl::access::fence_space::local_space);
        if( work_node == my_item.get_local_id()[0] ) {
            warp_work_node[warp_id] = WORK_GROUP_SIZE;
            warp_still_has_work[0] = false;
            my_work_left = 0;
        }
        my_item.barrier(sycl::access::fence_space::local_space);
        // if my warp has no work to-do, just keep waiting in
        // this while-loop (so that other warps don't deadlock
        //                  on the barriers)
        if(work_node >= WORK_GROUP_SIZE) {
            continue;
        }

        // Now work on the work_node's out-edges in batches of
        // size WARP_SIZE 
        size_t current_edge = group_first_edges[work_node] + my_warp_local_id,
                  last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            index_type dst_node = edge_dst[current_edge];
            if( node_level[dst_node] > LEVEL ) {
                node_level[dst_node] = LEVEL;
                made_updates[0] = true;
            }
            current_edge += WARP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run fine-grained-scheduling on an iteration of BFS
 *
 * Should only be called internally by BFS_iter.
 * Works on nodes with 
 * 0 < degree < MIN_WARP_SCHED_DEGREE
 * in a fine-grained fashion.
 * Sets my_work_left to 0 if my node got worked on.
 *
 * Barriers:
 *   - global and local at start and end.
 *
 * my_item: sycl object representing my item
 * my_work_left: the amount of work my item still wants done.
 *               May be modified.
 * my_first_edge: the edge index of the first out-edge from the node
 *                I want worked on, if any
 * my_first_edge: the edge index of the last out-edge from the node
 *                I want worked on, if any
 */
void BFS_iter::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
                                       index_type &my_work_left,
                                       index_type my_first_edge, 
                                       index_type my_last_edge)
{
    /// Setup /////////////////////////////////////////////////////////////////
    my_item.barrier(sycl::access::fence_space::global_and_local);
    // start with no fine-grained edges
    if(my_item.get_local_id()[0] == 0) {
        num_fine_grained_edges[0].store(0);
    }
    my_item.barrier(sycl::access::fence_space::local_space);
    // get my position in the line to have my fine-grained edges handled
    size_t my_fine_grained_index = INF;
    if(0 < my_work_left && my_work_left < MIN_WARP_SCHED_DEGREE) {
        my_fine_grained_index = num_fine_grained_edges[0].fetch_add(my_work_left);
    }
    // Set fine-grained edges to an invalid initial value
    for(size_t i = my_item.get_local_id()[0]; i < FINE_GRAINED_EDGE_CAP; i += WORK_GROUP_SIZE) {
        fine_grained_edges[i] = NEDGES;
    }
    ///////////////////////////////////////////////////////////////////////////
    my_item.barrier(sycl::access::fence_space::local_space);

    /// Work on fine-grained edges ////////////////////////////////////////////
    gpu_size_t total_work = num_fine_grained_edges[0].load();
    for(size_t i = 0; i < total_work; i += FINE_GRAINED_EDGE_CAP) {
        // If I have work to do and
        // my edges fit on the fine-grained edges array,
        // put my edges on the array!
        while(my_work_left > 0 && my_fine_grained_index < FINE_GRAINED_EDGE_CAP) {
            fine_grained_edges[my_fine_grained_index++] = my_first_edge++;
            my_work_left--;
        }
        // Wait for everyone's edges to get on the array
        my_item.barrier(sycl::access::fence_space::local_space);
        // Now, work on the fine-grained edges
        for(size_t j = my_item.get_local_id()[0]; j < FINE_GRAINED_EDGE_CAP; j += WORK_GROUP_SIZE) {
            // get the edge I'm supposed to work on and reset it to an invalid value
            // for next time
            index_type edge_index = fine_grained_edges[j];
            fine_grained_edges[j] = NEDGES;
            // If I got an invalid edge, skip
            if(edge_index >= NEDGES) {
                continue;
            }
            // If I got a valid edge, see if I can improve its BFS level
            index_type dst_node = edge_dst[edge_index];
            if( node_level[dst_node] > LEVEL ) {
                node_level[dst_node] = LEVEL;
                made_updates[0] = true;
            }
        }
        // Now we've done some amount of work, so I can lower my fine-grained index
        my_fine_grained_index -= FINE_GRAINED_EDGE_CAP;
        // Make sure the node level updates go through, and that
        // the resets of the array entries finalize
        my_item.barrier(sycl::access::fence_space::global_and_local);
    }
    ///////////////////////////////////////////////////////////////////////////

    my_item.barrier(sycl::access::fence_space::global_and_local);
}

/*
 * Perform an iteration of BFS at the given LEVEL
 * (i.e. looking for nodes with distance to source of LEVEL)
 */
void BFS_iter::operator()(sycl::nd_item<1> my_item) {
    // Get my global and local ids
    sycl::id<1> my_global_id = my_item.get_global_id(),
                my_local_id = my_item.get_local_id();
    // figure out what work I need to do (if any)
    index_type my_work_left = 0,
               my_first_edge = INF,
               my_last_edge = INF;
    if(my_global_id[0] < NNODES && node_level[my_global_id] == (LEVEL-1)) {
        // get my src node, first edge, and last edge in private memory
        my_first_edge = row_start[my_global_id],
        my_last_edge = row_start[my_global_id[0]+1];
        // put first/last edge into local memory
        group_first_edges[my_local_id] = my_first_edge;
        group_last_edges[my_local_id] = my_last_edge;
        // store the work I have left to do in private memory
        my_work_left = my_last_edge - my_first_edge;
    }

    // Work on nodes as a group
    group_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
    warp_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
    fine_grained_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
}
