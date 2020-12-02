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

// Figure out positions in out-worklist and update node levels
class BFSPreSweep {
    private: 
    // work distribution constants
    const size_t NNODES,
                 NEDGES,
                 LEVEL,
                 MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE,
                 // TODO : should we do any fine-grained scheduling?
                 MIN_WARP_SCHED_DEGREE = 1, //WARP_SIZE,
                 FINE_GRAINED_EDGE_CAP = WORK_GROUP_SIZE;
    // global memory:
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // read-access to the CSR graph
                       row_start,
                       edge_dst,
                       // in-worklist
                       in_worklist;
    sycl::accessor<size_t, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // number of items on the in_worklist
                       in_worklist_size;
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // read-write access to the BFS level of each node
                       node_level;
    sycl::accessor<gpu_size_t, 1,
                   sycl::access::mode::atomic,
                   sycl::access::target::global_buffer>
                       // number of nodes needed in out-worklist
                       // for each group
                       num_out_nodes;
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
    BFSPreSweep(size_t level,
                sycl::buffer<index_type, 1> &in_worklist_buf,
                sycl::buffer<size_t, 1> &in_worklist_size_buf,
                sycl::buffer<gpu_size_t, 1> &num_out_nodes_buf,
                SYCL_CSR_Graph &sycl_graph,
                sycl::handler &cgh)
        // initialize some constants
        : NNODES{ sycl_graph.nnodes }
        , NEDGES{ sycl_graph.nedges }
        , LEVEL{ level }
        // nodes we need to work on
        , in_worklist{ in_worklist_buf, cgh }
        , in_worklist_size{ in_worklist_size_buf, cgh }
        // number of nodes which each group will put in the out-worklist
        , num_out_nodes{ num_out_nodes_buf, cgh }
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


// Figure out the out-worklist
class BFSPostSweep {
    private: 
    sycl::stream sycl_stream;
    // work distribution constants
    const size_t NNODES,
                 NEDGES,
                 LEVEL,
                 MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE,
                 // TODO: Should we do any fine-grained scheduling?
                 MIN_WARP_SCHED_DEGREE = 1, //WARP_SIZE,
                 FINE_GRAINED_EDGE_CAP = WORK_GROUP_SIZE;
    // global memory:
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // read-access to the CSR graph
                       row_start,
                       edge_dst,
                       // in-worklist
                       in_worklist;
    sycl::accessor<index_type, 1,
                   sycl::access::mode::write,
                   sycl::access::target::global_buffer>
                       // out-worklist
                       out_worklist;
    sycl::accessor<size_t, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // number of items on the in_worklist
                       in_worklist_size;
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // read-write access to the BFS level of each node
                       node_level;
    sycl::accessor<gpu_size_t, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // number of nodes needed in out-worklist
                       // for each group
                       num_out_nodes;
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
                       num_fine_grained_edges,
                       // current spot in out-worklist
                       out_worklist_index;

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
    BFSPostSweep(size_t level,
                 sycl::buffer<index_type, 1> &in_worklist_buf,
                 sycl::buffer<size_t, 1> &in_worklist_size_buf,
                 sycl::buffer<gpu_size_t, 1> &num_out_nodes_buf,
                 sycl::buffer<index_type, 1> &out_worklist_buf,
                 SYCL_CSR_Graph &sycl_graph,
                 sycl::handler &cgh)
        // initialize some constants
        : NNODES{ sycl_graph.nnodes }
        , NEDGES{ sycl_graph.nedges }
        , LEVEL{ level }
        // nodes we need to work on
        , in_worklist{ in_worklist_buf, cgh }
        , in_worklist_size{ in_worklist_size_buf, cgh }
        // out nodes on a buffer
        , out_worklist{ out_worklist_buf, cgh }
        // number of nodes which each group will put in the out-worklist
        , num_out_nodes{ num_out_nodes_buf, cgh }
        // group-local memory
        , group_first_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
        , group_last_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
        , group_work_node{ sycl::range<1>{1}, cgh }
        , warp_work_node{ sycl::range<1>{WARPS_PER_GROUP}, cgh }
        , warp_still_has_work{ sycl::range<1>{1}, cgh }
        , fine_grained_edges{ sycl::range<1>{FINE_GRAINED_EDGE_CAP}, cgh }
        , num_fine_grained_edges{ sycl::range<1>{1}, cgh }
        , out_worklist_index{ sycl::range<1>{1}, cgh }
        // graph in global memory
        , row_start{ sycl_graph.row_start, cgh }
        , edge_dst{  sycl_graph.edge_dst,  cgh }
        , node_level{sycl_graph.node_data, cgh }
        , sycl_stream{1024, 256, cgh}
        { }

    void operator()(sycl::nd_item<1>);
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
                                     num_out_nodes_buf,
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
                                      num_out_nodes_buf,
                                      *out_worklist_buf,
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

/**
 * Run group-scheduling on a pre-sweep of BFS
 *
 * Should only be called internally.
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
void BFSPreSweep::group_scheduling(const sycl::nd_item<1> &my_item,
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
    size_t num_new_nodes = 0;
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
        index_type current_edge = group_first_edges[work_node] + my_item.get_local_id()[0],
                      last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            index_type dst_node = edge_dst[current_edge];
            if( node_level[dst_node] == INF ) {
                node_level[dst_node] = INF - my_item.get_local_id()[0] - 1;
                num_new_nodes++;
            }
            current_edge += WORK_GROUP_SIZE;
        }
    }
    if(num_new_nodes > 0) {
        num_out_nodes[my_item.get_group(0)].fetch_add(num_new_nodes);
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run warp-scheduling on an iteration of BFS
 *
 * Should only be called internally.
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
void BFSPreSweep::warp_scheduling(const sycl::nd_item<1> &my_item,
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
    size_t num_new_nodes = 0;
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
        index_type current_edge = group_first_edges[work_node] + my_warp_local_id,
                      last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            index_type dst_node = edge_dst[current_edge];
            if( node_level[dst_node] == INF ) {
                node_level[dst_node] = INF - my_item.get_local_id()[0] - 1;
                num_new_nodes++;
            }
            current_edge += WARP_SIZE;
        }
    }
    if(num_new_nodes > 0) {
        num_out_nodes[my_item.get_group(0)].fetch_add(num_new_nodes);
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run fine-grained-scheduling on an iteration of BFS
 *
 * Should only be called internally.
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
void BFSPreSweep::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
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
    size_t num_new_nodes = 0;
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
            if( node_level[dst_node] == INF ) {
                node_level[dst_node] = INF - 1;
                num_new_nodes++;
            }
        }
        // Now we've done some amount of work, so I can lower my fine-grained index
        my_fine_grained_index -= FINE_GRAINED_EDGE_CAP;
        // Make sure the node level updates go through, and that
        // the resets of the array entries finalize
        my_item.barrier(sycl::access::fence_space::global_and_local);
    }
    ///////////////////////////////////////////////////////////////////////////
    if(num_new_nodes > 0) {
        num_out_nodes[my_item.get_group(0)].fetch_add(num_new_nodes);
    }

    my_item.barrier(sycl::access::fence_space::global_and_local);
}

/*
 * Perform a preSweep of BFS at the given LEVEL
 * (i.e. looking for nodes with distance to source of LEVEL,
 *  and counting how many we find without updating their level yet)
 */
void BFSPreSweep::operator()(sycl::nd_item<1> my_item) {
    // Get my global and local ids
    sycl::id<1> my_global_id = my_item.get_global_id(),
                my_local_id = my_item.get_local_id();
    // clear the num-out-items for this group
    if(my_local_id[0] == 0) {
        num_out_nodes[my_item.get_group(0)].store(0);
    }
    my_item.barrier(sycl::access::fence_space::local_space);

    for(size_t i = (my_global_id[0] - my_local_id[0]);
        i < in_worklist_size[0];
        i += NUM_WORK_ITEMS)
    {
        // figure out what work I need to do (if any)
        index_type my_node = NNODES,
                   my_work_left = 0,
                   my_first_edge = INF,
                   my_last_edge = INF;
        size_t my_worklist_index = i + my_local_id[0];
        if(my_worklist_index < in_worklist_size[0]) {
            my_node = in_worklist[my_worklist_index];
            // get my src node, first edge, and last edge in private memory
            my_first_edge = row_start[my_node],
            my_last_edge = row_start[my_node+1];
            // put first/last edge into local memory
            group_first_edges[my_local_id] = my_first_edge;
            group_last_edges[my_local_id] = my_last_edge;
            // store the work I have left to do in private memory
            my_work_left = my_last_edge - my_first_edge;
        }

        // Work on nodes as a group
        group_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
        warp_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
        //fine_grained_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
    }
}

/// POST SWEEP IMPLEMENTATION ////////////////////////////////////////////////////////////////////////

/**
 * Run group-scheduling on a post-sweep of BFS
 *
 * Should only be called internally.
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
void BFSPostSweep::group_scheduling(const sycl::nd_item<1> &my_item,
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
        index_type current_edge = group_first_edges[work_node] + my_item.get_local_id()[0],
                      last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            index_type dst_node = edge_dst[current_edge];
            if( node_level[dst_node] == INF - my_item.get_local_id()[0] - 1) {
                node_level[dst_node] = LEVEL;
                out_worklist[out_worklist_index[0].fetch_add(1)] = dst_node;
            }
            current_edge += WORK_GROUP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run warp-scheduling on an post-sweep of BFS
 *
 * Should only be called internally.
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
void BFSPostSweep::warp_scheduling(const sycl::nd_item<1> &my_item,
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
        index_type current_edge = group_first_edges[work_node] + my_warp_local_id,
                      last_edge = group_last_edges[work_node];
        while( current_edge < last_edge ) {
            index_type dst_node = edge_dst[current_edge];
            if( node_level[dst_node] == INF - my_item.get_local_id()[0] - 1) {
                node_level[dst_node] = LEVEL;
                out_worklist[out_worklist_index[0].fetch_add(1)] = dst_node;
            }
            current_edge += WARP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run fine-grained-scheduling on a post-sweep of BFS
 *
 * Should only be called internally.
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
void BFSPostSweep::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
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
            if( node_level[dst_node] == INF - 1 ) {
                node_level[dst_node] = LEVEL;
                out_worklist[out_worklist_index[0].fetch_add(1)] = dst_node;
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
 * Perform a postSweep of BFS at the given LEVEL
 * (i.e. looking for nodes with distance to source of LEVEL,
 *  and counting how many we find without updating their level yet)
 */
void BFSPostSweep::operator()(sycl::nd_item<1> my_item) {
    // Get my global and local ids
    sycl::id<1> my_global_id = my_item.get_global_id(),
                my_local_id = my_item.get_local_id();

    // get out-worklist index
    if(my_local_id[0] == 0) {
        gpu_size_t ndx = 0;
        if(my_item.get_group(0) > 0) {
            ndx = num_out_nodes[my_item.get_group(0)-1];
        }
        out_worklist_index[0].store(ndx);
    }
    my_item.barrier(sycl::access::fence_space::local_space);
    // reset everything in my portion of the out-worklist to invalid
    for(size_t i = my_local_id[0] + out_worklist_index[0].load();
        i < num_out_nodes[my_item.get_group(0)];
        i += WORK_GROUP_SIZE)
    {
        out_worklist[i] = NNODES;
    }
    my_item.barrier(sycl::access::fence_space::local_space);

    for(size_t i = (my_global_id[0] - my_local_id[0]);
        i < in_worklist_size[0];
        i += NUM_WORK_ITEMS)
    {
        // figure out what work I need to do (if any)
        index_type my_node = NNODES,
                   my_work_left = 0,
                   my_first_edge = INF,
                   my_last_edge = INF;
        size_t my_worklist_index = i + my_local_id[0];
        if(my_worklist_index < in_worklist_size[0]) {
            // get my node from the worklist
            my_node = in_worklist[my_worklist_index];
            if(my_node < NNODES) {
                // get my src node, first edge, and last edge in private memory
                my_first_edge = row_start[my_node],
                my_last_edge = row_start[my_node+1];
                // put first/last edge into local memory
                group_first_edges[my_local_id] = my_first_edge;
                group_last_edges[my_local_id] = my_last_edge;
                // store the work I have left to do in private memory
                my_work_left = my_last_edge - my_first_edge;
            }
        }

        // Work on nodes as a group
        group_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
        warp_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
        //fine_grained_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
    }
}
