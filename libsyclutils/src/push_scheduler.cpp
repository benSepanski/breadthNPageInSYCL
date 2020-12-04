#include "push_scheduler.h"

/// Group Scheduling //////////////////////////////////////////////////////////
void PushScheduler::group_scheduling(const sycl::nd_item<1> &my_item,
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
        index_type current_edge = group_first_edges[work_node] + my_item.get_local_id()[0],
                      last_edge = group_last_edges[work_node],
                       src_node = group_src_nodes[work_node];
        while( current_edge < last_edge ) {
            applyPushOperator(src_node, current_edge);
            current_edge += WORK_GROUP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}
///////////////////////////////////////////////////////////////////////////////


/// Warp Scheduling ///////////////////////////////////////////////////////////
void PushOperator::warp_scheduling(const sycl::nd_item<1> &my_item,
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
                      last_edge = group_last_edges[work_node],
                       src_node = group_src_nodes[work_node];
        while( current_edge < last_edge ) {
            applyPushOperator(src_node, current_edge);
            current_edge += WARP_SIZE;
        }
    }
    my_item.barrier(sycl::access::fence_space::global_and_local);
}
///////////////////////////////////////////////////////////////////////////////


/// fine-grainedScheduling ////////////////////////////////////////////////////
void PushOperator::fine_grained_scheduling(const sycl::nd_item<1> &my_item,
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
    gpu_size_t my_fine_grained_index = INF;
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
            fine_grained_src_node[my_fine_grained_index] = my_src_node;
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
                        src_node  = fine_grained_src_node[j];
            fine_grained_edges[j] = NEDGES;
            // If I got an invalid edge, skip
            if(edge_index >= NEDGES) {
                continue;
            }
            // If I got a valid edge, apply!
            applyPushOperator(src_node, current_edge);
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
void PushScheduler::operator()(sycl::nd_item<1> my_item) {
    // Get my global and local ids
    sycl::id<1> my_global_id = my_item.get_global_id(),
                my_local_id = my_item.get_local_id();
    // now iterate through the worklist (making sure that if anyone
    //                                   in my group has work, then
    //                                   I join in to help)
    gpu_size_t wl_index = my_global_id[0];
    for(gpu_size_t i = my_global_id[0] - my_local_id[0];
        i < in_wl.getSize();
        i += NUM_WORK_ITEMS;)
    {
        // figure out what work I need to do (if any)
        index_type my_work_left, my_src_node, my_first_edge, my_last_edge;
        if(wl_index < in_wl.getSize()) {
            // get my src node, first edge, and last edge in private memory
            pop(wl_index, my_src_node);
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
    }
}
///////////////////////////////////////////////////////////////////////////////
