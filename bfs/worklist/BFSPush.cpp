#include "BFSPush.h"

/**
* Run group-scheduling on a push implementation of BFS
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
void BFSPush::group_scheduling(const sycl::nd_item<1> &my_item,
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
            applyToOutEdge(my_item, current_edge);
            current_edge += WORK_GROUP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run warp-scheduling on an push-implementation of BFS
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
void BFSPush::warp_scheduling(const sycl::nd_item<1> &my_item,
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
            applyToOutEdge(my_item, current_edge);
            current_edge += WARP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}


/**
 * Run fine-grained-scheduling on a push-implementaiton of BFS
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
void BFSPush::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
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
            // work on that edge
            applyToOutEdge(my_item, edge_index);
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
 * Perform a push-implementation of BFS at the given LEVEL
 * (i.e. looking for nodes with distance to source of LEVEL,
 *  and counting how many we find without updating their level yet)
 */
void BFSPush::operator()(sycl::nd_item<1> my_item) {
    // Get my global and local ids
    sycl::id<1> my_global_id = my_item.get_global_id(),
                my_local_id = my_item.get_local_id();

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
        fine_grained_scheduling(my_item, my_work_left, my_first_edge, my_last_edge);
    }
}
