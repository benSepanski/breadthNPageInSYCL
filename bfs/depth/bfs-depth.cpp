#include <chrono>
#include <iostream>
#include <limits>
#include <CL/cl.h>
#include <CL/sycl.hpp>

// From libsyclutils
//
// THREAD_BLOCK_SIZE WARP_SIZE
#include "kernel_sizing.h"
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

// classes used to name SYCL kernels
class bfs_init;
class update_level;
// One iteration of BFS
class BFS_iter{
    public:
    static const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
    private: 
    // work distribution constants
    const size_t NNODES,
                 NEDGES,
                 WARPS_PER_GROUP = (WORK_GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE,
                 MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE,
                 MIN_WARP_SCHED_DEGREE = WARP_SIZE,
                 FINE_GRAINED_EDGE_CAP = WORK_GROUP_SIZE,
                 // TODO: make this depend on local memory availability
                 GROUP_QUEUE_CAP = 4 * WORK_GROUP_SIZE;
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
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // read access to current level
                       LEVEL;
    sycl::accessor<bool, 1,
                   sycl::access::mode::write,
                   sycl::access::target::global_buffer>
                       // set to true iff we made updates
                       made_updates,
                       // true iff a node relaxed to level LEVEL-1
                       min_update_made;
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
                       fine_grained_edges,
                       // for a breadth-limited bfs (breadth/depth hybrid)
                       group_queue;
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local> 
                       // hold the bfs level of the current group-work node
                       group_work_level,
                       // hold the bfs level of the current work node for each warp
                       warp_work_level,
                       // store bfs level of each edge being handled at a fine-grain
                       fine_grained_edges_level;
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local> 
                       // communication between warps
                       warp_still_has_work,
                       // stop breadth-limited bfs when queue is full
                       group_queue_full;
    sycl::accessor<gpu_size_t, 1,
                   sycl::access::mode::atomic,
                   sycl::access::target::local>
                       // fine-grained scheduling queue size
                       num_fine_grained_edges,
                       // start and stop of group-local queue
                       group_queue_start,
                       group_queue_end;

    /// operator
    void relax_dst(index_type edge_index, node_data_type level);

    /// Various forms of scheduling during a bfs iteration
    void group_scheduling(const sycl::nd_item<1> &my_item,
                          index_type &my_work_left,
                          index_type my_node, 
                          index_type my_first_edge, 
                          index_type my_last_edge);

    void warp_scheduling(const sycl::nd_item<1> &my_item,
                         index_type &my_work_left,
                          index_type my_node, 
                         index_type my_first_edge, 
                         index_type my_last_edge);

    void fine_grained_scheduling(const sycl::nd_item<1> &my_item,
                                 index_type &my_work_left,
                                 index_type my_node, 
                                 index_type my_first_edge, 
                                 index_type my_last_edge);

    // TODO: remove this
    sycl::stream debug_stream;
    public:
    BFS_iter(sycl::buffer<node_data_type, 1> &level_buf,
             sycl::buffer<bool, 1> &made_updates_buf,
             sycl::buffer<bool, 1> &min_update_made_buf,
             SYCL_CSR_Graph &sycl_graph,
             sycl::handler &cgh)
        // initialize some constants
        : NNODES{ sycl_graph.nnodes }
        , NEDGES{ sycl_graph.nedges }
        , LEVEL { level_buf, cgh }
        // to record updates
        , made_updates{ made_updates_buf, cgh }
        , min_update_made{ min_update_made_buf, cgh }
        // group-local memory
        , group_first_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
        , group_last_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
        , group_work_node{ sycl::range<1>{1}, cgh }
        , group_work_level{ sycl::range<1>{1}, cgh }
        , warp_work_node{ sycl::range<1>{WARPS_PER_GROUP}, cgh }
        , warp_work_level{ sycl::range<1>{WARPS_PER_GROUP}, cgh }
        , warp_still_has_work{ sycl::range<1>{1}, cgh }
        , fine_grained_edges{ sycl::range<1>{FINE_GRAINED_EDGE_CAP}, cgh }
        , fine_grained_edges_level{ sycl::range<1>{FINE_GRAINED_EDGE_CAP}, cgh }
        , num_fine_grained_edges{ sycl::range<1>{1}, cgh }
        , group_queue{ sycl::range<1>{GROUP_QUEUE_CAP}, cgh }
        , group_queue_start{ sycl::range<1>{1}, cgh }
        , group_queue_end{ sycl::range<1>{1}, cgh }
        , group_queue_full{ sycl::range<1>{1}, cgh }
        // graph in global memory
        , row_start{ sycl_graph.row_start, cgh }
        , edge_dst{  sycl_graph.edge_dst,  cgh }
        , node_level{sycl_graph.node_data, cgh }
        // TODO: Remove this
        , debug_stream{1024, 512, cgh}
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

    const size_t WORK_GROUP_SIZE = BFS_iter::WORK_GROUP_SIZE,
                 NUM_WORK_GROUPS = (sycl_graph.nnodes + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE,
                 NUM_WORK_ITEMS = NUM_WORK_GROUPS * WORK_GROUP_SIZE;
    // get level set up on device
    node_data_type level = 1;
    sycl::buffer<node_data_type, 1> level_buf(&level, sycl::range<1>{1});
    // use this to know when we're done
    bool made_updates = false,
         made_updates_local_copy = made_updates;
    sycl::buffer<bool, 1> made_updates_buf(&made_updates, sycl::range<1>{1});
    // use this to know when we can increase the level
    bool min_update_made = false;
    sycl::buffer<bool, 1> min_update_made_buf(&min_update_made, sycl::range<1>{1});
    // How many jobs to submit at a time?
    const size_t BFS_BATCH_SIZE = 10;
    do {
        // Run BFS_BATCH_SIZE iterations of BFS
        try{
        for(size_t i = 0; i < BFS_BATCH_SIZE; ++i) {
            // run the iteration
            queue.submit([&] (sycl::handler &cgh) {
                BFS_iter current_iter(level_buf, made_updates_buf, min_update_made_buf, sycl_graph, cgh);
                cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                   sycl::range<1>{WORK_GROUP_SIZE}},
                                 current_iter);
            });
            // If no min_update was made, we can increment the level
            queue.submit([&] (sycl::handler &cgh) {
                auto min_update_made_acc = min_update_made_buf.get_access<sycl::access::mode::read>(cgh);
                auto level_acc = level_buf.get_access<sycl::access::mode::read_write>(cgh);
                cgh.single_task<class update_level>([=] () {
                    if(!min_update_made_acc[0]) {
                        level_acc[0]++;
                    }
                });
            });
        }
        //  (buffers only block to write-out on destruction, so we need this scope)
        {
            auto made_updates_acc = made_updates_buf.get_access<sycl::access::mode::read_write>();
            made_updates_local_copy = made_updates_acc[0];
            made_updates_acc[0] = false;
        } // End check for Updates SYCL scope
    } catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
        if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
        }
        std::exit(1);
    }
    } while(made_updates_local_copy);
    // Wait for BFS to finish and throw asynchronous errors if any
    queue.wait_and_throw();
}


int sycl_main(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    // Run sycl bfs in a try-catch block.
    try {
        sycl_bfs(sycl_graph, queue);
    } catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
        if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
        }
        std::exit(1);
    }

    return 0;
}

/**
 * Put a node on the group queue
 *
 * Should only be called internally.
 * Attempts to relax the destination of the given edge_index
 * down to the given level.
 *
 * Records that this iteration of BFS made updates
 *
 * If the group queue is not full, puts the dest node on the group queue.
 * If that fills up the group queue, marks the group queue as full.
 *
 * Performs no blocking synchronization.
 *
 * param edge_index: The edge to relax
 * param level: The level to relax down to (i.e. source node's level + 1)
 */
void BFS_iter::relax_dst(index_type edge_index, node_data_type level) {
    index_type dst_node = edge_dst[edge_index];
    if( node_level[dst_node] > level ) {
        node_level[dst_node] = level;
        made_updates[0] = true;
        if(level == LEVEL[0]-1) {
            min_update_made[0] = true;
        }
        // If there is room, add this node to the group-queue
        if(!group_queue_full[0]) {
            gpu_size_t group_queue_index = group_queue_end[0].fetch_add(1);
            // make sure indices wrap around
            group_queue_index %= GROUP_QUEUE_CAP;
            // since GROUP_QUEUE_CAP is a power of 2, logical and with
            // GROUP_QUEUE_CAP-1 is the same as modding by GROUP_QUEUE_CAP
            group_queue_end[0].fetch_and(GROUP_QUEUE_CAP-1);
            // if this filled up the queue, mark the queue as full
            if(group_queue_index + 1 == group_queue_start[0].load()) {
                group_queue_full[0] = true;
            }
            // Otherwise, put the node on the group queue
            else {
                group_queue[group_queue_index] = dst_node;
            }
        }
    }
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
 * my_node: the node I'm working on
 * my_first_edge: the edge index of the first out-edge from the node
 *                I want worked on, if any
 * my_first_edge: the edge index of the last out-edge from the node
 *                I want worked on, if any
 */
void BFS_iter::group_scheduling(const sycl::nd_item<1> &my_item,
                                index_type &my_work_left,
                                index_type my_node, 
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
        // Otherwise, copy the work node into private memory,
        // clear the group-work-node for next time,
        // and load the group-work-node's level into local memory
        index_type work_node = group_work_node[0];
        my_item.barrier(sycl::access::fence_space::local_space);
        if( work_node == my_item.get_local_id()[0] ) {
            group_work_node[0] = WORK_GROUP_SIZE;
            group_work_level[0] = node_level[my_node] + 1;
            my_work_left = 0;
        }
        my_item.barrier(sycl::access::fence_space::local_space);
        node_data_type work_level = group_work_level[0];

        // Now work on the work_node's out-edges in batches of
        // size WORK_GROUP_SIZE
        size_t current_edge = group_first_edges[work_node] + my_item.get_local_id()[0],
                  last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            relax_dst(current_edge, work_level);
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
 * my_node: the node I am working on
 * my_first_edge: the edge index of the first out-edge from the node
 *                I want worked on, if any
 * my_first_edge: the edge index of the last out-edge from the node
 *                I want worked on, if any
 */
void BFS_iter::warp_scheduling(const sycl::nd_item<1> &my_item,
                               index_type &my_work_left,
                               index_type my_node, 
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
        // Otherwise, copy the work node into private memory,
        // set warp_still_has_work to false for next time,
        // and put working level into local memory
        index_type work_node = warp_work_node[warp_id];
        my_item.barrier(sycl::access::fence_space::local_space);
        if( work_node == my_item.get_local_id()[0] ) {
            warp_work_node[warp_id] = WORK_GROUP_SIZE;
            warp_work_level[warp_id] = node_level[my_node] + 1;
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
        node_data_type work_level = warp_work_level[warp_id];

        // Now work on the work_node's out-edges in batches of
        // size WARP_SIZE 
        size_t current_edge = group_first_edges[work_node] + my_warp_local_id,
                  last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            relax_dst(current_edge, work_level);
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
 * my_node: the node I am working on
 * my_first_edge: the edge index of the first out-edge from the node
 *                I want worked on, if any
 * my_first_edge: the edge index of the last out-edge from the node
 *                I want worked on, if any
 */
void BFS_iter::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
                                       index_type &my_work_left,
                                       index_type my_node,
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
            fine_grained_edges[my_fine_grained_index] = my_first_edge++;
            fine_grained_edges_level[my_fine_grained_index++] = node_level[my_node] + 1;
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
            relax_dst(edge_index, fine_grained_edges_level[j]);
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
    // make sure group queue is empty
    if(my_local_id[0] == 0) {
        group_queue_full[0] = false;
        group_queue_start[0].store(0);
        group_queue_end[0].store(0);
    }
    my_item.barrier(sycl::access::fence_space::local_space);
    // Initialize group queue to the nodes of the current level or higher
    if(my_global_id[0] < NNODES
       && node_level[my_global_id] < INF 
       && node_level[my_global_id] >= LEVEL[0]-1)
    {
        group_queue[group_queue_end[0].fetch_add(1)] = my_global_id[0];
    }
    my_item.barrier(sycl::access::fence_space::local_space);

    while(group_queue_end[0].load() != group_queue_start[0].load()) {
        // get size of group-queue
        gpu_size_t group_queue_size = (group_queue_end[0].load() - group_queue_start[0].load()
                                       + GROUP_QUEUE_CAP) % GROUP_QUEUE_CAP;
        gpu_size_t my_group_queue_index = (group_queue_start[0].load() + my_local_id[0])
                                          % GROUP_QUEUE_CAP;
        my_item.barrier(sycl::access::fence_space::local_space);
        // Figure out if I have a node on the group-queue
        index_type my_node = NNODES,
                   my_work_left = 0,
                   my_first_edge = INF,
                   my_last_edge = INF;
        if(my_local_id[0] < group_queue_size) {
            my_node = group_queue[my_group_queue_index];
            // get my src node, first edge, and last edge in private memory
            my_first_edge = row_start[my_node],
            my_last_edge = row_start[my_node+1];
            // put first/last edge into local memory
            group_first_edges[my_local_id] = my_first_edge;
            group_last_edges[my_local_id] = my_last_edge;
            // store the work I have left to do in private memory
            my_work_left = my_last_edge - my_first_edge;
        }
        // Move the group-queue start forwards
        if(my_local_id[0] == 0) {
            group_queue_start[0].fetch_add(group_queue_size);
            // since GROUP_QUEUE_CAP is a power of 2, logical and with
            // GROUP_QUEUE_CAP-1 is the same as modding by GROUP_QUEUE_CAP
            group_queue_start[0].fetch_and(GROUP_QUEUE_CAP-1);
        }

        // Work on nodes as a group
        // BARRIER HERE FROM START OF GROUP_SCHEDULING
        group_scheduling(my_item, my_work_left, my_node, my_first_edge, my_last_edge);
        warp_scheduling(my_item, my_work_left, my_node, my_first_edge, my_last_edge);
        fine_grained_scheduling(my_item, my_work_left, my_node, my_first_edge, my_last_edge);
        // If the queue didn't fill up, keep going
        if(group_queue_full[0]) {
            break;
        }
        break;
    }
}
