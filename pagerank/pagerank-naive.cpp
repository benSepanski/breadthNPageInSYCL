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
class prob_init;
class prob_update;
class res_reset;

const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
             NUM_WORK_GROUPS = NUM_THREAD_BLOCKS,
             NUM_WORK_ITEMS  = NUM_WORK_GROUPS * WORK_GROUP_SIZE,
             WARPS_PER_GROUP = WORK_GROUP_SIZE / WARP_SIZE;

struct PROperatorInfo {
    sycl::accessor<float, 2,
                   sycl::access::mode::atomic,
                   sycl::access::target::global_buffer>
                       // residuals (separated by group) (NNODES, NUM_WORK_GROUP)
                       //   my group's portion of next residual goes in [node][group_nr]
                       residuals_by_group;
    sycl::accessor<float, 1,
                   sycl::access::mode::read,
                   sycl::access::target::global_buffer>
                       // previous residuals
                       prev_residuals;
    sycl::accessor<bool, 1,
                   sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
                       // Is this node on the out-worklist?
                       on_out_wl;
    sycl::accessor<int, 2,
                   sycl::access::mode::atomic,
                   sycl::access::target::global_buffer>
                       mutex;
    /** Called at start of push scheduling */
    void initialize(const sycl::nd_item<1> &my_item) { }

    /** Constructor **/
    PROperatorInfo( sycl::buffer<float, 2> &residuals_by_group_buf,
                    sycl::buffer<float, 1> &prev_residuals,
                    sycl::buffer<bool, 1> &on_out_wl_buf,
                    sycl::buffer<int, 2> &mutex_buf,
                    sycl::handler &cgh ) 
        : residuals_by_group{ residuals_by_group_buf, cgh }
        , prev_residuals{ prev_residuals, cgh }
        , on_out_wl{ on_out_wl_buf, cgh }
        , mutex{ mutex_buf, cgh }
    { }
    /** We must provide a copy constructor */
    PROperatorInfo( const PROperatorInfo &that )
        : residuals_by_group{ that.residuals_by_group }
        , prev_residuals{ that.prev_residuals }
        , on_out_wl{ that.on_out_wl }
        , mutex{ that.mutex }
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
        index_type dst_node = edge_dst[edge_index];
        sycl::id<2> target_id(dst_node, my_item.get_group(0));

        float update = opInfo.prev_residuals[src_node];
        // VIE FOR LOCK
        sycl::accessor<int, 2, sycl::access::mode::atomic, sycl::access::target::global_buffer> mut = opInfo.mutex;
        sycl::atomic<int> lock = mut[target_id];
        int unlocked = false;
        while(!lock.compare_exchange_strong(unlocked, true));
        // LOCK OBTAINED
        float prev = opInfo.residuals_by_group[target_id].load();
        opInfo.residuals_by_group[target_id].store(prev + update);
        my_item.mem_fence(sycl::access::fence_space::global_space);
        // UNLOCK
        opInfo.mutex[target_id].store(false);

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
    sycl::buffer<float, 1> prev_res_buf(sycl::range<1>{sycl_graph.nnodes});
    sycl::buffer<int, 2> mutex_buf(sycl::range<2>{sycl_graph.nnodes, NUM_WORK_GROUPS});

    // Build and initialize the worklist pipe
    //Pipe wl_pipe{(gpu_size_t) sycl_graph.nnodes,
    Pipe wl_pipe{(gpu_size_t) sycl_graph.nedges,
                 (gpu_size_t) sycl_graph.nnodes,
                 NUM_WORK_GROUPS};
    wl_pipe.initialize(queue);

    // Initialize probabilities to 0 and intiailize the first residuals to (1.0-ALPHA)
    // for each node.
    queue.submit([&] (sycl::handler &cgh) {
        // some constants
        const gpu_size_t NNODES = (gpu_size_t) sycl_graph.nnodes;
        const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
        const gpu_size_t NUM_WORK_ITEMS = NUM_WORK_GROUPS * WORK_GROUP_SIZE;
        // pr probabilities
        auto prob = P_CURR_buf.get_access<sycl::access::mode::discard_write>(cgh);
        // pr residuals
        auto res = res_buf.get_access<sycl::access::mode::discard_write>(cgh);
        auto mutex = mutex_buf.get_access<sycl::access::mode::discard_write>(cgh);
        // out-worklist
        InWorklist in_wl(wl_pipe, cgh);

        cgh.parallel_for<class prob_init>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                            sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // make sure in_wl has enough room
            if(my_item.get_local_id()[0] == 0) {
                in_wl.setSize(NNODES);
            }
            my_item.barrier(sycl::access::fence_space::global_space);
            // Add nodes to worklist and set initial probabilities
            for(index_type node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                prob[node] = 0.0;
                // put node on *node*th spot of in-worklist
                in_wl.push(node, node);
            }
            // Initialize the first residuals to 1.0 - alpha, and set the mutexes to unlocked
            for(index_type node = my_item.get_global_id()[0]; node < NNODES; node += NUM_WORK_ITEMS) {
                for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                    res[node][wg] = (1.0 - ALPHA) / NUM_WORK_GROUPS;
                    mutex[node][wg] = false;
                }
            }
        });
    });

    gpu_size_t in_wl_size = sycl_graph.nnodes;
    bool rerun = false,
         rerun_local_copy = false;
    sycl::buffer<bool, 1> rerun_buf(&rerun, sycl::range<1>{1});
    sycl::buffer<bool, 1> on_out_wl_buf(sycl::range<1>{sycl_graph.nnodes});
    // begin pagerank
    while(in_wl_size > 0 && ++iterations <= MAX_ITERATIONS) {
        std::cout << "iter: " << iterations << "\n"
                  << "in_wl_size: " << in_wl_size << std::endl;
        // If not re-running due to a full out-worklist...
        if(!rerun_local_copy) {
            // Update probabilities and reset residuals
            queue.submit([&](sycl::handler &cgh) {
                // In-worklist
                InWorklist in_wl(wl_pipe, cgh);
                // residual and probs
                auto res = res_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto prev_res = prev_res_buf.get_access<sycl::access::mode::discard_write>(cgh);
                auto probs = P_CURR_buf.get_access<sycl::access::mode::read_write>(cgh);
                // row starts
                auto row_start = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);
                // we want to reset on_out_wl_buf
                auto on_out_wl = on_out_wl_buf.get_access<sycl::access::mode::discard_write>(cgh);
                // update probs and reset residuals
                cgh.parallel_for<class prob_update>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                      sycl::range<1>{WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> my_item) {
                    for(size_t index = my_item.get_global_id()[0]; index < in_wl.getSize(); index += NUM_WORK_ITEMS) {
                        index_type node;
                        in_wl.pop(index, node);
                        // figure out total residual and add it to the prob
                        float total_residual = 0;
                        for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                            total_residual += res[node][wg];
                            res[node][wg] = 0;
                        }
                        probs[node] += total_residual;
                        // store the total residual, scaled appropriately, into the previous
                        // residual slot so the next iteration can use it for updates
                        index_type src_degree = row_start[node+1] - row_start[node];
                        prev_res[node] = total_residual * ALPHA / src_degree;
                        on_out_wl[node] = false;
                    }
                });
            });
        }
        // If re-running due to a full out-worklist, reset the new residuals
        else {
            queue.submit([&](sycl::handler &cgh) {
                // In-worklist
                InWorklist in_wl(wl_pipe, cgh);
                // residuals
                auto res = res_buf.get_access<sycl::access::mode::read_write>(cgh);
                // now reset them
                cgh.parallel_for<class res_reset>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                    sycl::range<1>{WORK_GROUP_SIZE}},
                [=](sycl::nd_item<1> my_item) {
                    for(size_t index = my_item.get_global_id()[0]; index < in_wl.getSize(); index += NUM_WORK_ITEMS) {
                        index_type node;
                        in_wl.pop(index, node);
                        for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                            res[node][wg] = 0;
                        }
                    }
                });
            });
        }
        queue.wait_and_throw();
        // Run an iteration of pagerank
        std::cout << "Running PR Iteration" << std::endl;
        queue.submit([&](sycl::handler &cgh) {
            PROperatorInfo prInfo( res_buf, prev_res_buf, on_out_wl_buf, mutex_buf, cgh );
            PRIter currentIter( sycl_graph, wl_pipe, cgh, rerun_buf, prInfo );
            cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                               sycl::range<1>{WORK_GROUP_SIZE}},
                             currentIter);
        });
        // compress out-worklist
        queue.wait_and_throw();
        wl_pipe.compress(queue);
        /*
        queue.submit([&](sycl::handler &cgh) {
            sycl::stream strm(1024, 512, cgh);
            InWorklist in_wl(wl_pipe, cgh);
            OutWorklist out_wl(wl_pipe, cgh);
            cgh.single_task<class DEBUG>([=]() {
                in_wl.print(strm);
                out_wl.print(strm);
            });
        });
        */
        queue.wait_and_throw();
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
    }
}
