#include <climits>
#include <iostream>
#include <CL/sycl.hpp>

// SYCL_CSR_Graph node_data_type index_type
#include "sycl_csr_graph.h"
// Pipe
#include "pipe.h"
// OutWorklist
#include "out_worklist.h"
// InWorklist
#include "in_worklist.h"
// PushScheduler
#include "push_scheduler.h"

extern const float ALPHA = 0.85;
extern const float EPSILON = 0.000001;
extern int MAX_ITERATIONS ;
int iterations = 0 ;

namespace sycl = cl::sycl;

// class names for SYCL kernels
class init;
class prob_update;
class InitOnOutWL;
class HardReset;

extern size_t num_work_groups;

struct PROperatorInfo {
    // global accessors
    sycl::accessor<float, 2,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // residuals (separated by group) (NNODES, NUM_WORK_GROUP)
                       //   my group's portion of next residual goes in [node][group_nr]
                       residuals_by_group;
    sycl::accessor<float, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // update coming out of this node
                       outgoing_update;
    sycl::accessor<size_t, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       mutex;
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       on_out_wl;
    // local memory
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::local>
                       bids_made;
    /** Called at start of push scheduling */
    void initialize(const sycl::nd_item<1> &my_item) { }

    /** Constructor **/
    PROperatorInfo( sycl::buffer<float, 2> &residuals_by_group_buf,
                    sycl::buffer<float, 1> &outgoing_update_buf,
                    sycl::buffer<bool, 1> &on_out_wl_buf,
                    sycl::buffer<size_t, 1> &mutex_buf,
                    sycl::handler &cgh ) 
        : residuals_by_group{ residuals_by_group_buf, cgh }
        , outgoing_update{ outgoing_update_buf, cgh }
        , on_out_wl{ on_out_wl_buf, cgh }
        , mutex{ mutex_buf, cgh }
        , bids_made{ sycl::range<1>{1}, cgh }
    { }
    /** We must provide a copy constructor */
    PROperatorInfo( const PROperatorInfo &that )
        : residuals_by_group{ that.residuals_by_group }
        , outgoing_update{ that.outgoing_update }
        , on_out_wl{ that.on_out_wl }
        , mutex{ that.mutex }
        , bids_made{ that.bids_made }
    { }
};


// Define our PR push operator
class PRIter : public PushScheduler<PRIter, PROperatorInfo> {
    public:
    PRIter(gpu_size_t num_work_groups,
           SYCL_CSR_Graph &sycl_graph, Pipe &pipe, sycl::handler &cgh,
           sycl::buffer<bool, 1> &out_worklist_needs_compression,
           PROperatorInfo &opInfo)
        : PushScheduler{num_work_groups, sycl_graph, pipe, cgh, out_worklist_needs_compression, opInfo}
        { }

    // Do a page-rank update
    void applyPushOperator(const sycl::nd_item<1> &my_item,
                           index_type src_node,
                           index_type edge_index) 
    {
        // Get my dest node if my edge is valid
        index_type dst_node = NNODES;
        if(edge_index < NEDGES && src_node < NNODES) {
            dst_node = edge_dst[edge_index];
        }
        // make sure bids_made starts out as false
        opInfo.bids_made[0] = false;
        my_item.barrier(sycl::access::fence_space::local_space);
        // Sit around til everyone in my group gets a chance to perform
        // their update!
        while(true) {
            // If I have stuff to do, compete for my mutex
            if(edge_index < NEDGES && dst_node < NNODES && src_node < NNODES) {
                opInfo.mutex[dst_node] = my_item.get_global_id()[0];
                opInfo.bids_made[0] = true;
            }
            // wait for bids to finalize...
            my_item.barrier();
            // If nobody has work to do, great! we're done.
            if(!opInfo.bids_made[0]) {
                break;
            }
            my_item.barrier();
            opInfo.bids_made[0] = false;
            // If I have stuff to do and got my mutex, do my work!
            if(edge_index < NEDGES && dst_node < NNODES && src_node < NNODES
               && opInfo.mutex[dst_node] == my_item.get_global_id()[0]) 
            {
                // do my update
                float update = opInfo.outgoing_update[src_node];
                float prev = opInfo.residuals_by_group[dst_node][my_item.get_group(0)];
                opInfo.residuals_by_group[dst_node][my_item.get_group(0)] += update;
                // See if I need to push the dst node
                if(   prev < EPSILON / NUM_WORK_GROUPS
                   && prev + update >= EPSILON / NUM_WORK_GROUPS
                   && !opInfo.on_out_wl[dst_node]) 
                {
                    bool push_success = out_wl.push(dst_node);
                    if(push_success) {
                        opInfo.on_out_wl[dst_node] = true;
                    }
                    else {
                        out_worklist_full[0] = true;
                    }
                }
                // I got to do my work! so I'm done
                edge_index = NEDGES;
                dst_node = NNODES;
                src_node = NNODES;
            }
            // Now wait for everyone's updates to finalize
            my_item.barrier();
        }
    }
};

// probability of each node as computed by pagerank
float *P_CURR;

// declaration of pagerank function, which wil be called by main
void sycl_pagerank(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue);

int sycl_main(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    std::cerr << "NUM WORK GROUPS: " << num_work_groups << "\n";
    try {
        sycl_pagerank(sycl_graph, queue);
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
        if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
        }
        std::exit(1);
    }
                
   return 0;
}


void sycl_pagerank(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                 NUM_WORK_GROUPS = num_work_groups,
                 NUM_WORK_ITEMS  = NUM_WORK_GROUPS * WORK_GROUP_SIZE,
                 WARPS_PER_GROUP = WORK_GROUP_SIZE / WARP_SIZE;
    // build buffers for probability and probability residuals
    P_CURR = (float*) calloc(sycl_graph.nnodes, sizeof(float));
    assert(P_CURR != NULL);
    sycl::buffer<float, 1> P_CURR_buf(P_CURR, sycl::range<1>{sycl_graph.nnodes});
    sycl::buffer<float, 2> res_buf(sycl::range<2>{sycl_graph.nnodes, NUM_WORK_GROUPS});
    sycl::buffer<float, 1> outgoing_update_buf(sycl::range<1>{sycl_graph.nnodes});
    sycl::buffer<size_t, 1> mutex_buf(sycl::range<1>{sycl_graph.nnodes});

    // Build and initialize the worklist pipe (the max
    // is needed for small graphs so that no group runs out of space
    // on its portion of the out-worklist)
    Pipe wl_pipe{sycl::max(sycl::max((gpu_size_t) sycl_graph.nedges, (gpu_size_t) NUM_WORK_ITEMS), (gpu_size_t) sycl_graph.nnodes),
                 (gpu_size_t) sycl_graph.nnodes,
                 (gpu_size_t) NUM_WORK_GROUPS};
    wl_pipe.initialize(queue);

    // Initialize probabilities to 1-ALPHA,
    // residuals to 0, and outgoing updates to alpha*(1-alpha)/src_degree
    // for each node.
    //
    // Also, put each node on the in-worklist
    queue.submit([&] (sycl::handler &cgh) {
        // some constants
        const gpu_size_t NNODES = (gpu_size_t) sycl_graph.nnodes;
        // pr probabilities
        auto prob = P_CURR_buf.get_access<sycl::access::mode::write>(cgh);
        auto res = res_buf.get_access<sycl::access::mode::write>(cgh);
        auto outgoing_update = outgoing_update_buf.get_access<sycl::access::mode::write>(cgh);
        // out-worklist and graph
        InWorklist in_wl(wl_pipe, cgh);
        auto row_start = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<class init>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                       sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // make sure in_wl has enough room
            if(my_item.get_local_id()[0] == 0) {
                in_wl.setSize(NNODES);
            }
            my_item.barrier(sycl::access::fence_space::global_space);
            // Add nodes to worklist and set initial probabilities
            for(index_type node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                prob[node] = 1.0-ALPHA;
                // put node on *node*th spot of in-worklist
                in_wl.push(node, node);
            }
            // Initialize the residuals to 0
            for(index_type node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                    res[node][wg] = 0.0;
                }
            }
            // Initialize out-going updates to alpha*(1-alpha)/degree
            for(index_type node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                outgoing_update[node] = ALPHA * (1-ALPHA) / (row_start[node+1] - row_start[node]);
            }
    }); });
    // Initially, nobody is on the out-worklist
    sycl::buffer<bool, 1> on_out_wl_buf(sycl::range<1>{sycl_graph.nnodes});
    queue.submit([&] (sycl::handler &cgh) {
        auto on_out_wl = on_out_wl_buf.get_access<sycl::access::mode::write>(cgh);
        const size_t NNODES = sycl_graph.nnodes;
        cgh.parallel_for<class InitOnOutWL>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                              sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            for(size_t i = my_item.get_global_id()[0]; i < NNODES; i += NUM_WORK_ITEMS) {
                on_out_wl[i] = false;
            }
    }); });


    size_t num_kernel_reruns = 0;
    // local copy of in-worklist size
    gpu_size_t in_wl_size = sycl_graph.nnodes;
    // Used by PushScheduler to tell if you need to retry.
    bool rerun = false, rerun_host_copy = false;
    sycl::buffer<bool, 1> rerun_buf(&rerun, sycl::range<1>{1});
    // begin pagerank
    while(in_wl_size > 0 && ++iterations <= MAX_ITERATIONS) {
        // Run an iteration of pagerank
        queue.submit([&](sycl::handler &cgh) {
            PROperatorInfo prInfo( res_buf, outgoing_update_buf, on_out_wl_buf, mutex_buf, cgh );
            PRIter currentIter(NUM_WORK_GROUPS, sycl_graph, wl_pipe, cgh, rerun_buf, prInfo );
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             currentIter);
        });
        // Do we need to re-run?
        {
            auto rerun_acc = rerun_buf.get_access<sycl::access::mode::read>();
            rerun_host_copy = rerun_acc[0];
            num_kernel_reruns += rerun_host_copy;
        }
        // If not re-running,
        // Update probabilities and reset residuals and outgoing updates.
        if(!rerun_host_copy) {
            // Swap slots
            wl_pipe.compress(queue);
            wl_pipe.swapSlots(queue);
            // update probs, reset residuals, get outgoing updates, and reset
            // on_out_wl
            queue.submit([&](sycl::handler &cgh) {
                // graph and worklists
                const size_t NNODES = sycl_graph.nnodes;
                const size_t NEDGES = sycl_graph.nedges;
                auto row_start = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);
                InWorklist in_wl(wl_pipe, cgh);
                // residual, updates, and probs
                auto res = res_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto outgoing_update = outgoing_update_buf.get_access<sycl::access::mode::write>(cgh);
                auto probs = P_CURR_buf.get_access<sycl::access::mode::read_write>(cgh);
                // clear who is on the out-wl
                auto on_out_wl = on_out_wl_buf.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<class prob_update>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                      sycl::range<1>{WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> my_item) {
                    index_type node;
                    for(size_t index = my_item.get_global_id()[0]; index < in_wl.getSize(); index += NUM_WORK_ITEMS) {
                        // pop *index*th entry of in-worklist into node
                        in_wl.pop(index, node);
                        // figure out total residual and add it to the prob
                        float total_residual = 0;
                        for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                            total_residual += res[node][wg];
                            res[node][wg] = 0;
                        }
                        probs[node] += total_residual;
                        // store the total residual, scaled appropriately, for
                        // future updates
                        index_type src_degree = row_start[node+1] - row_start[node];
                        outgoing_update[node] = total_residual * ALPHA / src_degree;
                    }
                    // clear on_out_wl
                    for(size_t node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                        on_out_wl[node] = false;
                    }
            }); });
            // Get in-worklist size (inside a new scope so that the
            //                       host accessor gets destroyed)
            {
                sycl::buffer<gpu_size_t, 1> in_wl_size_buf = wl_pipe.get_in_worklist_size_buf();
                auto in_wl_size_acc = in_wl_size_buf.get_access<sycl::access::mode::read>();
                in_wl_size = in_wl_size_acc[0];
            }
        }
        // If re-running, clear residuals and compress
        else {
            queue.submit([&](sycl::handler &cgh) {
                auto res = res_buf.get_access<sycl::access::mode::write>(cgh);
                const size_t NNODES = sycl_graph.nnodes;
                cgh.parallel_for<class HardReset>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                    sycl::range<1>{WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> my_item) {
                    for(size_t i = my_item.get_global_id()[0]; i < NNODES; i += NUM_WORK_ITEMS) {
                        for(gpu_size_t j = 0; j < NUM_WORK_GROUPS; ++j) {
                            res[i][j] = 0.0;
                    } }
            }); });
            wl_pipe.compress(queue);
        }
    }
    queue.wait_and_throw();
    std::cerr << "NUM KERNEL RERUNS: " << num_kernel_reruns << "\n";
}
