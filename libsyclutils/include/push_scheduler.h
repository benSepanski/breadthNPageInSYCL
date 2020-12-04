#include <CL/sycl.hpp>
//
// SYCL_CSR_Graph node_data_type index_type
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

#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32

// "derive" from this class using the
// curiously recurring template pattern as described in
// https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/limitations
template <class PushOperator>
class PushScheduler {
    protected:
    const gpu_size_t NNODES,
                     NEDGES,
                     // TODO: MAke these variable
                     WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                     NUM_WORK_GROUPS = 6,
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
    sycl::accessor<node_data_type, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // read-write access to the data of node
                       node_data;
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

    /**
     * Apply the push operator along an edge
     *
     * src_node: the source node of the edge
     * current_edge: the edge index
     */
    void applyPushOperator(index_type src_node, index_type current_edge) {
        static_cast<PushOperator&>(*this).applyPushOperator();
    };


    public:
        PushScheduler(SYCLGraph &sycl_graph, Pipe &pipe, sycl::handler &cgh) 
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
            , node_data{ sycl_graph.node_data, cgh }
            // group-local memory
            , group_src_nodes  { sycl::range<1>{NUM_WORK_GROUPS}, cgh }
            , group_first_edges{ sycl::range<1>{NUM_WORK_GROUPS}, cgh }
            , group_last_edges { sycl::range<1>{NUM_WORK_GROUPS}, cgh }
            , group_work_node{ sycl::range<1>{1}, cgh}
            , warp_work_node { sycl::range<1>{1}, cgh}
            , fine_grained_edges{ sycl::range<1>{FINE_GRAINED_EDGE_CAPACITY}, cgh }
            , warp_still_has_work{ sycl::range<1>{1}, cgh }
            , num_fine_grained_edges{ sycl::range<1>{1}, cgh }
        { }

        // SYCL Kernel
        void operator()(sycl::nd_item<1>);
};

#endif
