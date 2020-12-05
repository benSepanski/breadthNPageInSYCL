#include <CL/sycl.hpp>
//
// THREAD_BLOCK_SIZE WARP_SIZE NUM_THREAD_BLOCKS
#include "kernel_sizing.h"
// SYCL_CSR_Graph index_type
#include "sycl_csr_graph.h"
// Pipe
#include "pipe.h"
// InWorklist
#include "in_worklist.h"
// OutWorklist
#include "out_worklist.h"

extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

#ifndef BREADTHNPAGEINSYCL_LIBSYCLUTILS_PUSHSCHEDULER_
#define BREADTHNPAGEINSYCL_LIBSYCLUTILS_PUSHSCHEDULER_

// "derive" from this class using the
// curiously recurring template pattern as described in
// https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/limitations
//
// The OperatorInfo class can be used by the PushOperator as a way of carrying
// extra data. The PushScheduler will use the copy constructor to
// make its own copy.
// The PushScheduler calls OperatorInfo's initialize(nd_item<1>) method at
// the beginning of scheduling.
//
template <class PushOperator, class OperatorInfo>
class PushScheduler {
    protected:
    const gpu_size_t NNODES,
                     NEDGES,
                     // TODO: MAke these variable
                     WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                     NUM_WORK_GROUPS = NUM_THREAD_BLOCKS,
                     NUM_WORK_ITEMS = NUM_WORK_GROUPS * WORK_GROUP_SIZE,
                     WARPS_PER_GROUP = (WORK_GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE,
                     // TODO: make these variable, and make sure
                     //       fine-grained edge capacity is dependent on
                     //       the available memory
                     MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE,
                     MIN_WARP_SCHED_DEGREE = WARP_SIZE,
                     FINE_GRAINED_EDGE_CAPACITY = WORK_GROUP_SIZE;
    // worklists
    InWorklist in_wl;
    OutWorklist out_wl;
    // global SYCL memory:
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // read-access to the CSR graph
                       row_start,
                       edge_dst;
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // Did any group have to stop working because of a full worklist?
                       out_worklist_needs_compression;
    // group-local memory:
    sycl::accessor<index_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local> 
                       // local memory for nodes to store their first/last edges and the src node
                       group_src_nodes,
                       group_first_edges,
                       group_last_edges,
                       // local memory for a work-node during group-scheduling
                       group_work_node,
                       // local memory for work-nodes during warp-scheduling
                       warp_work_node,
                       // fine-grained scheduling for edges
                       fine_grained_src_nodes,
                       fine_grained_edges;
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local> 
                       // communication between warps
                       warp_still_has_work,
                       // my portion of the out-worklist is full.
                       // Must be set by derived classes.
                       out_worklist_full;
    sycl::accessor<gpu_size_t, 1,
                   sycl::access::mode::atomic,
                   sycl::access::target::local>
                       // fine-grained scheduling queue size
                       num_fine_grained_edges;
    // Operator-specific information
    OperatorInfo opInfo;

    // scheduling methods
    /**
     * Run group-scheduling using a push operator
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
     */
    void group_scheduling(const sycl::nd_item<1> &my_item,
                          index_type &my_work_left);

    /**
     * Run warp-scheduling using a push operator
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
     */
    void warp_scheduling(const sycl::nd_item<1> &my_item,
                         index_type &my_work_left);

    /**
     * Run fine-grained-scheduling on a push operator
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
     * my_src_node: the source node of my edges
     * my_first_edge: the edge index of the first out-edge from the node
     *                I want worked on, if any
     */
    void fine_grained_scheduling(const sycl::nd_item<1> &my_item,
                                 index_type &my_work_left,
                                 index_type my_src_node,
                                 index_type my_first_edge);

    public:
        PushScheduler(SYCL_CSR_Graph &sycl_graph, Pipe &pipe, sycl::handler &cgh,
                      sycl::buffer<bool, 1> &out_worklist_needs_compression_buf,
                      OperatorInfo &operatorInfo) 
            // This cast is okay because sycl_driver does a check
            // TODO: we should probably also do a check here though for safety?
            : NNODES{ (gpu_size_t) sycl_graph.nnodes }
            , NEDGES{ (gpu_size_t) sycl_graph.nedges }
            // in/out worklists
            , in_wl{ pipe, cgh }
            , out_wl{ pipe, cgh }
            // CSR Graph in memory
            , row_start{ sycl_graph.row_start, cgh }
            , edge_dst { sycl_graph.edge_dst , cgh }
            , out_worklist_needs_compression{ out_worklist_needs_compression_buf, cgh }
            // group-local memory
            , group_src_nodes  { sycl::range<1>{WORK_GROUP_SIZE}, cgh }
            , group_first_edges{ sycl::range<1>{WORK_GROUP_SIZE}, cgh }
            , group_last_edges { sycl::range<1>{WORK_GROUP_SIZE}, cgh }
            , group_work_node{ sycl::range<1>{1}, cgh}
            , warp_work_node { sycl::range<1>{WARPS_PER_GROUP}, cgh}
            , fine_grained_src_nodes{ sycl::range<1>{FINE_GRAINED_EDGE_CAPACITY}, cgh }
            , fine_grained_edges    { sycl::range<1>{FINE_GRAINED_EDGE_CAPACITY}, cgh }
            , warp_still_has_work{ sycl::range<1>{1}, cgh }
            , out_worklist_full  { sycl::range<1>{1}, cgh }
            , num_fine_grained_edges{ sycl::range<1>{1}, cgh }
            // operator-specific information
            , opInfo{ operatorInfo }
        { }

    // SYCL Kernel
    void operator()(sycl::nd_item<1>);

    /**
     * Apply the push operator along an edge
     *
     * Note: it not guaranteed that all threads call this function
     *       each time it is called
     *
     * my_item: my sycl work-item 
     * src_node: the source node of the edge. not defined if edge index
     *           is invalid.
     * current_edge: the edge index, or an invalid edge index (>= NEDGES)
     */
    void applyPushOperator(const sycl::nd_item<1> &my_item,
                           index_type src_node,
                           index_type current_edge) 
    {
        static_cast<PushOperator&>(*this).applyPushOperator(my_item,
                                                            src_node,
                                                            current_edge);
    };
};

/// Group Scheduling //////////////////////////////////////////////////////////
template <class PushOperator, class OperatorInfo>
void PushScheduler<PushOperator, OperatorInfo>::group_scheduling(const sycl::nd_item<1> &my_item,
                                                                 index_type &my_work_left)
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
        // (and for any global updates to stick)
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
        index_type   first_edge = group_first_edges[work_node],
                      last_edge = group_last_edges[work_node],
                       src_node = group_src_nodes[work_node],
                   current_edge = first_edge + my_item.get_local_id()[0];
        // Make sure every worker in the group enters the function call
        // or no worker in the group enters the function call
        while( current_edge - my_item.get_local_id()[0] < last_edge ) {
            applyPushOperator(my_item, src_node, current_edge);
            if(current_edge < last_edge) {
                current_edge += WORK_GROUP_SIZE;
            }
            else {
                current_edge = NEDGES;
            }
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}
///////////////////////////////////////////////////////////////////////////////


/// Warp Scheduling ///////////////////////////////////////////////////////////
template <class PushOperator, class OperatorInfo>
void PushScheduler<PushOperator, OperatorInfo>::warp_scheduling(const sycl::nd_item<1> &my_item,
                                                                index_type &my_work_left)
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
        // (and for any global updates to stick)
        my_item.barrier(sycl::access::fence_space::global_and_local);
        // If no-one competed for control of the group, we're done!
        if(!warp_still_has_work[0]) {
            break;
        }
        // We want every worker in the group to enters the function call
        // or no worker in the group enters the function call, so we need
        // to know how many times to enter the loop
        index_type niters = 0;
        for(gpu_size_t i = 0; i < WARPS_PER_GROUP; ++i) {
            index_type work = 0;
            if(warp_work_node[i] != WORK_GROUP_SIZE) {
                work = group_last_edges[warp_work_node[i]] - group_first_edges[warp_work_node[i]];
            }
            niters = (work > niters) ? work : niters;
        }
        // copy the work node into private memory
        // and set warp_still_has_work to false for next time
        index_type work_node = warp_work_node[warp_id];
        my_item.barrier(sycl::access::fence_space::local_space);
        if( work_node == my_item.get_local_id()[0] ) {
            warp_work_node[warp_id] = WORK_GROUP_SIZE;
            warp_still_has_work[0] = false;
            my_work_left = 0;
        }
        my_item.barrier(sycl::access::fence_space::local_space);
        // Now work on the work_node's out-edges in batches of
        // size WARP_SIZE 
        index_type first_edge = INF,
                    last_edge = INF,
                     src_node = INF,
                 current_edge = INF;
        if(work_node < WORK_GROUP_SIZE) {
            first_edge = group_first_edges[work_node],
            last_edge = group_last_edges[work_node],
            src_node = group_src_nodes[work_node],
            current_edge = first_edge + my_warp_local_id;
        }

        for( size_t i = 0; i < niters; ++i) {
            applyPushOperator(my_item, src_node, current_edge);
            if(current_edge < last_edge) {
                current_edge += WARP_SIZE;
            }
            else {
                current_edge = NEDGES;
            }
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}
///////////////////////////////////////////////////////////////////////////////


/// fine-grainedScheduling ////////////////////////////////////////////////////
template <class PushOperator, class OperatorInfo>
void PushScheduler<PushOperator, OperatorInfo>::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
                                                                        index_type &my_work_left,
                                                                        index_type my_src_node,
                                                                        index_type my_first_edge)
{
    /// Setup /////////////////////////////////////////////////////////////////
    my_item.barrier(sycl::access::fence_space::global_and_local);
    // start with no fine-grained edges
    if(my_item.get_local_id()[0] == 0) {
        num_fine_grained_edges[0].store(0);
    }
    my_item.barrier(sycl::access::fence_space::local_space);
    // get my position in the line to have my fine-grained edges handled
    gpu_size_t my_fine_grained_index = FINE_GRAINED_EDGE_CAPACITY;
    if(0 < my_work_left && my_work_left < MIN_WARP_SCHED_DEGREE) {
        my_fine_grained_index = num_fine_grained_edges[0].fetch_add(my_work_left);
    }
    // Set fine-grained edges to an invalid initial value
    for(gpu_size_t i = my_item.get_local_id()[0]; i < FINE_GRAINED_EDGE_CAPACITY; i += WORK_GROUP_SIZE) {
        fine_grained_edges[i] = NEDGES;
    }
    ///////////////////////////////////////////////////////////////////////////
    my_item.barrier(sycl::access::fence_space::local_space);

    /// Work on fine-grained edges ////////////////////////////////////////////
    gpu_size_t total_work = num_fine_grained_edges[0].load();
    for(gpu_size_t i = 0; i < total_work; i += FINE_GRAINED_EDGE_CAPACITY) {
        // If I have work to do and
        // my edges fit on the fine-grained edges array,
        // put my edges on the array!
        while(my_work_left > 0 && my_fine_grained_index < FINE_GRAINED_EDGE_CAPACITY) {
            fine_grained_edges[my_fine_grained_index] = my_first_edge++;
            fine_grained_src_nodes[my_fine_grained_index] = my_src_node;
            my_fine_grained_index++;
            my_work_left--;
        }
        // Wait for everyone's edges to get on the array
        my_item.barrier(sycl::access::fence_space::local_space);
        // Now, work on the fine-grained edges
        for(gpu_size_t j = my_item.get_local_id()[0]; j < FINE_GRAINED_EDGE_CAPACITY; j += WORK_GROUP_SIZE) {
            // get the edge I'm supposed to work on and reset it to an invalid value
            // for next time
            index_type edge_index = fine_grained_edges[j],
                        src_node = fine_grained_src_nodes[j];
            fine_grained_edges[j] = NEDGES;
            // If I got a valid edge, apply!
            applyPushOperator(my_item, src_node, edge_index);
        }
        // Now we've done some amount of work, so I can lower my fine-grained index
        my_fine_grained_index -= FINE_GRAINED_EDGE_CAPACITY;
        // Make sure the node level updates go through, and that
        // the resets of the array entries finalize
        my_item.barrier(sycl::access::fence_space::global_and_local);
    }
    ///////////////////////////////////////////////////////////////////////////

    my_item.barrier(sycl::access::fence_space::global_and_local);
}
///////////////////////////////////////////////////////////////////////////////


/// SYCL Kernel //////////////////////////////////////////////////////////////
template <class PushOperator, class OperatorInfo>
void PushScheduler<PushOperator, OperatorInfo>::operator()(sycl::nd_item<1> my_item) {
    // Get my global and local ids
    sycl::id<1> my_global_id = my_item.get_global_id(),
                my_local_id = my_item.get_local_id();
    if(my_local_id[0] == 0) {
        out_wl.initializeLocalMemory(my_item);
        out_worklist_full[0] = false;
    }
    // Initialize operator info
    opInfo.initialize(my_item);
    my_item.barrier(sycl::access::fence_space::global_and_local);
    // now iterate through the worklist (making sure that if anyone
    //                                   in my group has work, then
    //                                   I join in to help)
    gpu_size_t wl_index = my_global_id[0];
    for(gpu_size_t i = my_global_id[0] - my_local_id[0];
        i < in_wl.getSize();
        i += NUM_WORK_ITEMS)
    {
        // figure out what work I need to do (if any)
        index_type my_work_left, my_src_node, my_first_edge, my_last_edge;
        if(wl_index < in_wl.getSize()) {
            // get my src node, first edge, and last edge in private memory
            in_wl.pop(wl_index, my_src_node);
            my_first_edge = row_start[my_src_node],
            my_last_edge = row_start[my_src_node+1];
            // put first/last edge and src node into local memory
            group_src_nodes[my_local_id] = my_src_node;
            group_first_edges[my_local_id] = my_first_edge;
            group_last_edges[my_local_id] = my_last_edge;
            // store the work I have left to do in private memory
            my_work_left = my_last_edge - my_first_edge;
        }
        else {
            my_work_left = 0,
            my_src_node = INF,
            my_first_edge = INF,
            my_last_edge = INF;
        }
        wl_index += NUM_WORK_ITEMS;
        // Work on nodes as a group
        group_scheduling(my_item, my_work_left);
        warp_scheduling(my_item, my_work_left);
        fine_grained_scheduling(my_item, my_work_left, my_src_node, my_first_edge);
        // break if out-worklist is full
        if(out_worklist_full[0]) {
            break;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
    if(my_local_id[0] == 0) {
        out_wl.publishLocalMemory(my_item);
        if(out_worklist_full[0]) {
            out_worklist_needs_compression[0] = true;
        }
    }
}
///////////////////////////////////////////////////////////////////////////////

#endif
