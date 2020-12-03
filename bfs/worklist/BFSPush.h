#include <CL/sycl.hpp>

// From libsyclutils
//
// SYCL_CSR_Graph node_data_type index_type
#include "sycl_csr_graph.h"

#ifndef BREADTHNPAGEINSYCL_BFS_WORKLIST_BFSPUSH_
#define BREADTHNPAGEINSYCL_BFS_WORKLIST_BFSPUSH

// NVIDIA target can't do 64-bit atomic adds, even though it says it can // sycl_driver makes sure this is big enough
typedef uint32_t gpu_size_t;

#define WARP_SIZE 32

extern const size_t WORK_GROUP_SIZE,
                    NUM_WORK_GROUPS,
                    NUM_WORK_ITEMS,
                    WARPS_PER_GROUP;
extern const uint64_t INF;

/**
 * An abstract class which applies operator
 * repeatedly using a push-based BFS.
 */
class BFSPush {
    protected:
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
    BFSPush(size_t level,
            sycl::buffer<index_type, 1> &in_worklist_buf,
            sycl::buffer<size_t, 1> &in_worklist_size_buf,
            SYCL_CSR_Graph &sycl_graph,
            sycl::handler &cgh)
        // initialize some constants
        : NNODES{ sycl_graph.nnodes }
        , NEDGES{ sycl_graph.nedges }
        , LEVEL{ level }
        // nodes we need to work on
        , in_worklist{ in_worklist_buf, cgh }
        , in_worklist_size{ in_worklist_size_buf, cgh }
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

    virtual void applyToOutEdge(sycl::nd_item<1> my_item,
                                index_type edge_index);
};

#endif
