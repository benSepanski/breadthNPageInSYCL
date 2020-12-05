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
class init_on_out_wl;
class prob_update;
class res_reset;

const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
             NUM_WORK_GROUPS = NUM_THREAD_BLOCKS,
             NUM_WORK_ITEMS  = NUM_WORK_GROUPS * WORK_GROUP_SIZE,
             WARPS_PER_GROUP = WORK_GROUP_SIZE / WARP_SIZE;

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
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // Is this node on the out-worklist?
                       on_out_wl;
    sycl::accessor<size_t, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       mutex;
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
    PRIter(SYCL_CSR_Graph &sycl_graph, Pipe &pipe, sycl::handler &cgh,
           sycl::buffer<bool, 1> &out_worklist_needs_compression,
           PROperatorInfo &opInfo)
        : PushScheduler{sycl_graph, pipe, cgh, out_worklist_needs_compression, opInfo}
        { }

    void applyPushOperator(const sycl::nd_item<1> &my_item,
                           index_type src_node,
                           index_type edge_index) 
    {
        // Get my dest node if my edge is valid
        index_type dst_node = NNODES;
        if(edge_index < NEDGES) {
            dst_node = edge_dst[edge_index];
        }
        // make sure bids_made starts out as false
        opInfo.bids_made[0] = false;
        my_item.barrier(sycl::access::fence_space::local_space);
        // Sit around til everyone in my group gets a chance to perform
        // their update!
        while(true) {
            // If I have stuff to do, compete for my mutex
            if(edge_index < NEDGES) {
                opInfo.mutex[dst_node] = my_item.get_global_id()[0];
                opInfo.bids_made[0] = true;
            }
            // wait for bids to finalize...
            my_item.barrier(sycl::access::fence_space::global_and_local);
            // If nobody has work to do, great! we're done.
            if(!opInfo.bids_made[0]) {
                break;
            }
            opInfo.bids_made[0] = false;
            // If I have stuff to do and got my mutex, do my work!
            if(edge_index < NEDGES && opInfo.mutex[dst_node] == my_item.get_global_id()[0]) {
                sycl::id<2> target_id(dst_node, my_item.get_group(0));
                float prev = opInfo.residuals_by_group[target_id],
                      update = opInfo.outgoing_update[src_node];
                opInfo.residuals_by_group[target_id] += update;
                // if we don't already have this on the queue and we raised it above
                // the threshold that it may need to be worked on next time, put
                // the destination node on the queue!
                bool was_below_eps = (prev < EPSILON / NUM_WORK_GROUPS),
                     now_above_eps = (prev + update >= EPSILON / NUM_WORK_GROUPS);
                if(was_below_eps && now_above_eps && !opInfo.on_out_wl[dst_node]) {
                    bool push_success = out_wl.push(dst_node);
                    if(!push_success) {
                        out_worklist_full[0] = true;
                    }
                    else {
                        opInfo.on_out_wl[dst_node] = true;
                    }
                }
                // I got to do my work! so I'm done
                edge_index = NEDGES;
                dst_node = NNODES;
            }
            // Now wait for everyone's updates to finalize
            my_item.barrier(sycl::access::fence_space::global_and_local);
        }
    }
};


// probability of each node as computed by pagerank
float *P_CURR;

// declaration of pagerank function, which wil be called by main
void sycl_pagerank(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue);

int sycl_main(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
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
    // build buffers for probability and probability residuals
    P_CURR = new float[sycl_graph.nnodes];
    sycl::buffer<float, 1> P_CURR_buf(P_CURR, sycl::range<1>{sycl_graph.nnodes});
    sycl::buffer<float, 2> res_buf(sycl::range<2>{sycl_graph.nnodes, NUM_WORK_GROUPS});
    sycl::buffer<float, 1> outgoing_update_buf(sycl::range<1>{sycl_graph.nnodes});
    sycl::buffer<size_t, 1> mutex_buf(sycl::range<1>{sycl_graph.nnodes});

    // Build and initialize the worklist pipe
    //Pipe wl_pipe{(gpu_size_t) sycl_graph.nnodes,
    Pipe wl_pipe{(gpu_size_t) sycl_graph.nedges,
                 (gpu_size_t) sycl_graph.nnodes,
                 NUM_WORK_GROUPS};
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
        auto prob = P_CURR_buf.get_access<sycl::access::mode::discard_write>(cgh);
        auto res = res_buf.get_access<sycl::access::mode::discard_write>(cgh);
        auto outgoing_update = outgoing_update_buf.get_access<sycl::access::mode::discard_write>(cgh);
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
        });
    });

    // local copy of in-worklist size
    gpu_size_t in_wl_size = sycl_graph.nnodes;
    // does out-worklist needs to compress and re-try the iteration?
    bool rerun = false, rerun_local_copy = false;
    sycl::buffer<bool, 1> rerun_buf(&rerun, sycl::range<1>{1});
    // which nodes are already on the out-worklist buffer (if
    // we are retrying the iteration, we need to get new nodes on)
    sycl::buffer<bool, 1> on_out_wl_buf(sycl::range<1>{sycl_graph.nnodes});
    // make sure on_out_wl_buf starts out as false
    queue.submit([&](sycl::handler &cgh) {
        const size_t NNODES = sycl_graph.nnodes;
        auto on_out_wl = on_out_wl_buf.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class init_on_out_wl>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                 sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            for(size_t node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                on_out_wl[node] = false;
            }
    });});

    // begin pagerank
    while(in_wl_size > 0 && ++iterations <= MAX_ITERATIONS) {
        // Run an iteration of pagerank
        queue.submit([&](sycl::handler &cgh) {
            PROperatorInfo prInfo( res_buf, outgoing_update_buf, on_out_wl_buf, mutex_buf, cgh );
            PRIter currentIter( sycl_graph, wl_pipe, cgh, rerun_buf, prInfo );
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             currentIter);
        });
        // compress out-worklist
        wl_pipe.compress(queue);
        // Either setup for a re-run, or swap slots and get new in-worklist size
        {
            auto rerun_acc = rerun_buf.get_access<sycl::access::mode::read_write>();
            rerun_local_copy = rerun_acc[0];
            if(rerun_acc[0]) {
                iterations--;
                rerun_acc[0] = false;
            }
            else {
                wl_pipe.swapSlots(queue);
                sycl::buffer<gpu_size_t, 1> in_wl_size_buf = wl_pipe.get_in_worklist_size_buf();
                auto in_wl_size_acc = in_wl_size_buf.get_access<sycl::access::mode::read>();
                in_wl_size = in_wl_size_acc[0];
            }
        }
        // If not re-running due to a full out-worklist...
        if(!rerun_local_copy) {
            // Update probabilities and reset residuals and outgoing updates
            queue.submit([&](sycl::handler &cgh) {
                // graph
                const size_t NNODES = sycl_graph.nnodes;
                auto row_start = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);
                // residual, updates, and probs
                auto res = res_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto outgoing_update = outgoing_update_buf.get_access<sycl::access::mode::discard_write>(cgh);
                auto probs = P_CURR_buf.get_access<sycl::access::mode::read_write>(cgh);
                // nobody is on the queue anymore
                auto on_out_wl = on_out_wl_buf.get_access<sycl::access::mode::discard_write>(cgh);

                cgh.parallel_for<class prob_update>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                      sycl::range<1>{WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> my_item) {
                    for(size_t node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
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
                        on_out_wl[node] = false;
                    }
            }); });
        }
        // If re-running due to a full out-worklist, reset the residuals
        else {
            queue.submit([&](sycl::handler &cgh) {
                const size_t NNODES = sycl_graph.nnodes;
                auto res = res_buf.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<class res_reset>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                    sycl::range<1>{WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> my_item) {
                    for(size_t node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                        for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                            res[node][wg] = 0.0;
                    } }
            }); });
        }
    }
}
