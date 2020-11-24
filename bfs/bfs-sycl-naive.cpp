#include <iostream>
#include <limits>
#include <CL/cl.h>
#include <CL/sycl.hpp>

// From libsyclutils
//
// Host_CSR_Graph index_type
#include "host_csr_graph.h"
// SYCL_CSR_Graph
#include "sycl_csr_graph.h"

// Do this to save some typing
namespace sycl = cl::sycl;

// from support.cpp
extern index_type start_node;

typedef uint64_t node_data_type;
// gpu can't do 64-bit atomics, so we use 32-bits for sizes etc.
typedef uint32_t gpu_size_t;
// defined here, but const so need to declare as extern so support.cpp
// can use it
extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

// TODO: Make these extern consts which are determined by sycl_driver.cpp
#define NUM_THREAD_BLOCKS 4
#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32

// classes used to name SYCL kernels
class bfs;

/**
 * Given a sycl-graph, run bfs using
 * the given queue, storing the levels in node_data and using start_node
 * as the start node
 */ 
void sycl_bfs(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    // Run BFS on the device
    try {
        // submit to queue (recording profiling information in event) ///////////////////////////////////////
        sycl::event bfs_event = queue.submit([&] (sycl::handler &cgh) {
            /// Get constants needed on device ////////////////////////////////////////////////////
            //
            // We will need the start node
            const index_type START_NODE = start_node;
            // this is fine bc sycl_driver makes sure it fits in 32-bits
            const gpu_size_t NNODES = sycl_graph.nnodes;
            // Constants for work distribution
            const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                             NUM_WORK_GROUPS = NUM_THREAD_BLOCKS,
                             WARPS_PER_GROUP = (WORK_GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE,
                      // size of the out-worklist cache in local memory
                      GROUP_LOCAL_QUEUE_CAPACITY = 16 * THREAD_BLOCK_SIZE,
                      // minimum degree to use group-level scheduling
                      MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE + 1;
            ///////////////////////////////////////////////////////////////////////////////////////
            //
            /// Get accessors needed on device ////////////////////////////////////////////////////
            //
            // We need to be able to read the graph's edges
            auto row_start  = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);
            auto edge_dst   = sycl_graph.edge_dst .get_access<sycl::access::mode::read>(cgh);
            // and read/write to its node data
            auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::discard_write>(cgh);
            // We need to have "in" and "out" worklists
            sycl::buffer<index_type, 1> worklist1_buf(sycl::range<1>{sycl_graph.nnodes}),
                                            worklist2_buf(sycl::range<1>{sycl_graph.nnodes});
            auto even_in_worklist = worklist1_buf.get_access<sycl::access::mode::discard_write>(cgh),
                 odd_in_worklist  = worklist2_buf.get_access<sycl::access::mode::discard_write>(cgh);
            // We'll need an in-worklist size,  an out-worklist size, and a record
            // of how much of the in-worklist we've processed
            sycl::buffer<gpu_size_t, 1>  in_worklist_size_buf(sycl::range<1>{1}),
                                         out_worklist_size_buf(sycl::range<1>{1}),
                                         in_worklist_done_buf(sycl::range<1>{1});
            sycl::buffer<gpu_size_t, 1> level_complete_buf(sycl::range<1>{1});
            auto  in_worklist_size  = in_worklist_size_buf.get_access<sycl::access::mode::discard_write>(cgh);
            auto  out_worklist_size = out_worklist_size_buf.get_access<sycl::access::mode::atomic>(cgh),
                  in_worklist_done  = in_worklist_done_buf.get_access<sycl::access::mode::atomic>(cgh);
            auto level_complete = level_complete_buf.get_access<sycl::access::mode::atomic>(cgh);
            // We need to give each group a group-local queue
            sycl::accessor<index_type, 1, sycl::access::mode::read_write, sycl::access::target::local>
                group_local_queue(sycl::range<1>{GROUP_LOCAL_QUEUE_CAPACITY}, cgh);
            sycl::accessor<gpu_size_t, 1, sycl::access::mode::atomic, sycl::access::target::local>
                group_local_queue_size(sycl::range<1>{1}, cgh);
            // We also want to give each group space to peform warp-level scheduling
            sycl::accessor<index_type, 1, sycl::access::mode::read_write, sycl::access::target::local>
                warp_level_work_node(sycl::range<1>{WARPS_PER_GROUP}, cgh),
                warp_level_first_edge(sycl::range<1>{WARPS_PER_GROUP}, cgh),
                warp_level_last_edge(sycl::range<1>{WARPS_PER_GROUP}, cgh),
                warp_level_offset(sycl::range<1>{WARPS_PER_GROUP}, cgh);
            ///////////////////////////////////////////////////////////////////////////////////////
            //
            /// Now start the BFS job /////////////////////////////////////////////////////////////
            cgh.parallel_for_work_group<class bfs>(sycl::range<1>{NUM_WORK_GROUPS}, sycl::range<1>{WORK_GROUP_SIZE},
            [=](sycl::group<1> my_group) {
                /// Initialize node data, in-worklist, and out-worklist //////////////////
                my_group.parallel_for_work_item([&](sycl::h_item<1> my_item) {
                    size_t my_global_id = my_item.get_global_id()[0];
                    if(my_global_id < NNODES) {
                        node_level[my_global_id] = (my_global_id == START_NODE) ? 0 : INF;
                    }
                });
                in_worklist_size[0] = 1;
                in_worklist_done[0].store(0);
                out_worklist_size[0].store(0);
                odd_in_worklist[0] = START_NODE;
                // Allocate some private memory on each work item
                sycl::private_memory<index_type> my_node(my_group),
                                                 my_first_edge(my_group),
                                                 my_last_edge(my_group),
                                                 my_work_left(my_group);
                /////////////////////////////////////////////////////////////////////////
                //
                /// Now begin BFS Loop //////////////////////////////////////////////////
                level_complete[0].store(0);
                size_t level = 1;
                size_t in_worklist_block_offset = my_group.get_id()[0] * WORK_GROUP_SIZE;
                bool block_waiting_for_work = false;
                while(level < INF && in_worklist_size[0] > 0) {
                    /// wait until this block has work to do //////////////////
                    if( in_worklist_block_offset >= in_worklist_size[0] ) {
                        block_waiting_for_work = true;
                    }
                    while(block_waiting_for_work) {
                        if( level_complete[0].load() == level ) {
                            block_waiting_for_work = false;
                            level++;
                            in_worklist_block_offset = my_group.get_id()[0] * WORK_GROUP_SIZE;
                        }
                    }
                    if(in_worklist_size[0] == 0) {
                        break;
                    }

                    ///////////////////////////////////////////////////////////
                    //
                    /// Have each node figure out the work it wants done //////
                    my_group.parallel_for_work_item([&](sycl::h_item<1> my_item) {
                        size_t my_node_index = in_worklist_block_offset + my_item.get_local_id()[0];
                        if(my_node_index < in_worklist_size[0]) {
                            if( level % 2 == 0) {
                                my_node(my_item) = even_in_worklist[my_node_index];
                            }
                            else {
                                my_node(my_item) = odd_in_worklist[my_node_index];
                            }
                            my_first_edge(my_item) = row_start[my_node(my_item)];
                            my_last_edge(my_item) = row_start[my_node(my_item)+1];
                            my_work_left(my_item) = my_last_edge(my_item) - my_first_edge(my_item);
                        }
                        else {
                            my_node(my_item) = INF;
                            my_first_edge(my_item) = INF;
                            my_last_edge(my_item) = INF;
                            my_work_left(my_item) = 0;
                        }
                    });
                    ///////////////////////////////////////////////////////////
                    //
                    /// do group-level scheduling until no-one wants to ///////
                    index_type work_node, first_edge, last_edge, degree, work_done;
                    while(true) {
                        // Compete for control of the work_node
                        work_node = NNODES;
                        my_group.parallel_for_work_item([&](sycl::h_item<1> my_item) {
                            // If I want a lot of work done, bid for control of the work_node
                            if(my_work_left(my_item) >= MIN_GROUP_SCHED_DEGREE) {
                                work_node = my_node(my_item);
                            }
                        });
                        // If nobody competed for control of the node, we're done!
                        if(work_node == NNODES) {
                            break;
                        }
                        // Have the work_node controller move their information from
                        // private memory to group-local memory
                        my_group.parallel_for_work_item([&](sycl::h_item<1> my_item) {
                            if(my_node(my_item) == work_node) {
                                first_edge = my_first_edge(my_item);
                                last_edge = my_last_edge(my_item);
                                my_work_left(my_item) = 0;
                            }
                        });
                        // Now process the work_node's edges in batches of size WORK_GROUP_SIZE
                        degree = last_edge - first_edge;
                        for(work_done = 0; work_done < degree; work_done += WORK_GROUP_SIZE) {
                            // process WORK_GROUP_SIZE edges (using group-local queue)
                            my_group.parallel_for_work_item([&](sycl::h_item<1> my_item) {
                                index_type edge_index = first_edge + work_done + my_item.get_local_id()[0];
                                if(edge_index < last_edge) {
                                    index_type dst_node = edge_dst[edge_index];
                                    if( node_level[dst_node] == INF ) {
                                        node_level[dst_node] = level;
                                        gpu_size_t gl_index = group_local_queue_size[0].fetch_add(1);
                                        group_local_queue[gl_index] = dst_node;
                                    }
                                }
                            });
                            // if group-local queue is near full (or this is the last iteration)
                            // empty it into the global queue
                            gpu_size_t cur_gl_queue_size = group_local_queue_size[0].load();
                            if(   cur_gl_queue_size + WORK_GROUP_SIZE >= GROUP_LOCAL_QUEUE_CAPACITY
                                || work_done >= degree)
                            {
                                gpu_size_t cur_glob_queue_size = out_worklist_size[0].fetch_add(cur_gl_queue_size);
                                group_local_queue_size[0].store(0);
                                // copy group-local queue onto global queue
                                my_group.parallel_for_work_item(sycl::range<1>{cur_glob_queue_size},
                                [&](sycl::h_item<1> my_item) {
                                    size_t my_local_id = my_item.get_local_id()[0];
                                    if( level % 2 == 0) {
                                        odd_in_worklist[cur_glob_queue_size + my_local_id] = group_local_queue[my_local_id];
                                    }
                                    else {
                                        even_in_worklist[cur_glob_queue_size + my_local_id] = group_local_queue[my_local_id];
                                    }
                                });
                            }
                        }
                    }
                    ///////////////////////////////////////////////////////////
                    //
                    // Move block to next offset
                    in_worklist_block_offset += WORK_GROUP_SIZE * NUM_WORK_GROUPS;
                    gpu_size_t prev_work_done = in_worklist_done[0].fetch_add(1);
                    // If no more work to be done on this level, complete the level!
                    if(prev_work_done * WORK_GROUP_SIZE + WORK_GROUP_SIZE >= in_worklist_size[0]) {
                        in_worklist_done[0].store(0);
                        in_worklist_size[0] = out_worklist_size[0].load();
                        out_worklist_size[0].store(0);
                        my_group.mem_fence();
                        level_complete[0].fetch_add(1);
                    }
                }
                /////////////////////////////////////////////////////////////////////////
            });
            ///////////////////////////////////////////////////////////////////////////////////////
        });
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        cl_ulong submit_time = bfs_event.get_profiling_info<sycl::info::event_profiling::command_submit>(),
                 start_time  = bfs_event.get_profiling_info<sycl::info::event_profiling::command_start>(),
                 end_time    = bfs_event.get_profiling_info<sycl::info::event_profiling::command_end>();
        fprintf(stderr,
                "BFS Submit To Start = %0.2f ms\nBFS_Submit Start To End = %0.2f ms\n",
                (start_time - submit_time)/1000000.0,
                (end_time - start_time)/1000000.0);
    } catch (sycl::exception const& e) {
       std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
       if(e.get_cl_code() != CL_SUCCESS) {
           std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
       }
       std::exit(1);
    }

    // Wait for BFS to finish and handle asynchronous erros as in
    // https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
    try {
        queue.wait_and_throw();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
        }
        std::exit(1);
    }
}


int sycl_main(Host_CSR_Graph &host_graph, cl::sycl::queue &queue) {

   // copy start_node into local variable so we can use it inside SYCL kernels
   const index_type START_NODE = start_node;

   // Begin SYCL Scope 
   {
      // Build our sycl graph inside scope so that buffers can be destroyed
      // by destructor
      SYCL_CSR_Graph sycl_graph(&host_graph);
      // run bfs
      sycl_bfs(sycl_graph, queue);
   } // End SYCL scope

   return 0;
}


/*
              // put the bfs iter on the command queue
              cgh.parallel_for_work_group<class bfs_iter>(cl::sycl::range<1>{NUM_WORK_GROUPS},
                                                          cl::sycl::range<1>{WORK_GROUP_SIZE},
                [=] (cl::sycl::group<1> my_group) {
                    // Give each work-item private memory in which to store its node
                    cl::sycl::private_memory<index_type> my_node(my_group),
                                                         my_first_edge(my_group),
                                                         my_last_edge(my_group),
                                                         my_work_left(my_group);
                    /// Have each work-item figure out the node it needs to work on (if any) ////////////////
                    my_group.parallel_for_work_item( [&] (cl::sycl::h_item<1> local_item) {
                        if(local_item.get_global_id()[0] < in_worklist_size) {
                            size_t id = local_item.get_global_id()[0];
                            my_node(local_item) = in_worklist_acc[id];
                            // THESE TWO CALLS RETURNS AN LVALUE REFERENCE!
                            my_first_edge(local_item) = row_start_acc[id];
                            my_last_edge(local_item) = row_start_acc[id+1];
                            my_work_left(local_item) = my_last_edge(local_item) - my_first_edge(local_item);
                        }
                        else {
                            my_node(local_item) = WORK_GROUP_SIZE;
                            my_first_edge(local_item) = INF;
                            my_last_edge(local_item) = INF;
                            my_work_left(local_item) = 0;
                        }
                    });
                    /////////////////////////////////////////////////////////////////////////////////////////
                    //
                    // initialize work_node to an invalid group id
                    size_t work_node = WORK_GROUP_SIZE, prev_work_node;
                    // Build an atomic group-local queue size and set it to zero
                    cl::sycl::multi_ptr<uint32_t, cl::sycl::access::address_space::local_space> group_local_queue_size_ptr;
                    cl::sycl::atomic<uint32_t, cl::sycl::access::address_space::local_space> at_group_local_queue_size(group_local_queue_size_ptr);
                    at_group_local_queue_size.store(0);
                    /// Now do group-level scheduling (work on one node at a time as a group) ///////////////
                    while(true) {
                        prev_work_node = work_node;
                        /// Compete for the work node /////////////////////////////////////////////
                        my_group.parallel_for_work_item( [&] (cl::sycl::h_item<1> local_item) {
                            if( my_work_left(local_item) >= MIN_GROUP_SCHED_DEGREE) {
                                work_node = my_node(local_item);
                            }
                        });
                        ///////////////////////////////////////////////////////////////////////////
                        //
                        // If no-one competed for the work-node, we're done!
                        if(prev_work_node == work_node) {
                            break;
                        }
                        //
                        // Figure out the index range of the out-edges the work_node
                        index_type first_edge, last_edge, degree, first_edge_copy; 
                        my_group.parallel_for_work_item([&] (cl::sycl::h_item<1> local_item) {
                            if(work_node == my_node(local_item)) {
                                first_edge = my_first_edge(local_item);
                                last_edge  = my_last_edge(local_item);
                                my_work_left(local_item) = 0;
                            }
                        });
                        degree = last_edge - first_edge;
                        first_edge_copy = first_edge;
                        /// now work on the edges in batches of size WORK_GROUP_SIZE ////
                        while(first_edge_copy < last_edge) {
                            // figure out batch size
                            size_t batch_size = WORK_GROUP_SIZE;
                            if(batch_size > last_edge - first_edge_copy) {
                                batch_size = last_edge - first_edge_copy;
                            }
                            /// Work on this batch of edges ///////////////////
                            my_group.parallel_for_work_item(cl::sycl::range<1>{batch_size},
                            [&] (cl::sycl::h_item<1> local_item) {
                                index_type edge_index = first_edge_copy + local_item.get_local_id()[0];
                                index_type edge_dst = edge_dst_acc[edge_index];  // THIS RETURNS AN LVALUE REFERENCE!
                                if( node_data_acc[edge_dst] == INF ) {
                                    node_data_acc[edge_dst] = LEVEL;
                                    uint32_t group_local_queue_size = at_group_local_queue_size.fetch_add(1);
                                    group_local_queue_acc[group_local_queue_size] = edge_dst;
                                }
                            });
                            ///////////////////////////////////////////////////
                            //
                            first_edge_copy += batch_size;
                            /// Move group-local queue to global if full //////
                            uint32_t group_local_queue_size = at_group_local_queue_size.load();
                            if(group_local_queue_size > GROUP_LOCAL_QUEUE_SIZE - WORK_GROUP_SIZE || first_edge_copy >= last_edge) {
                                at_group_local_queue_size.store(0);
                                uint32_t offset = out_worklist_size_acc[0].fetch_add(group_local_queue_size);
                                my_group.parallel_for_work_item(cl::sycl::range<1>{group_local_queue_size},
                                [&] (cl::sycl::h_item<1> local_item) {
                                    size_t id = local_item.get_local_id()[0];
                                    out_worklist_acc[offset + id] = group_local_queue_acc[id];
                                });
                            }
                            ///////////////////////////////////////////////////
                        }
                        /////////////////////////////////////////////////////////////////
                        //
                        ///////////////////////////////////////////////////////////////////////////
                    }
                    /////////////////////////////////////////////////////////////////////////////////////////
                    //
                    /// Now do warp-level scheduling (work on one node at a time as a warp) /////////////////
                    //
                    // store id of warp and id inside warp
                    cl::sycl::private_memory<size_t> warp_id(my_group), warp_local_id(my_group);
                    my_group.parallel_for_work_item([&] (cl::sycl::h_item<1> local_item) {
                        warp_id(local_item) = local_item.get_local_id()[0] / WARP_SIZE;
                        warp_local_id(local_item) = local_item.get_local_id()[0] % WARP_SIZE;
                    });
                    while(true) {
                        /// Compete for the warp-level work node //////////////////////////////////
                        bool work_left = false;
                        my_group.parallel_for_work_item( [&] (cl::sycl::h_item<1> local_item) {
                            if( my_work_left(local_item) > 0) {
                                // bid for control
                                warp_level_work_node_acc[warp_id(local_item)] = warp_local_id(local_item);
                                work_left = true;
                            }
                        });
                        ///////////////////////////////////////////////////////////////////////////
                        //
                        // If no-one competed for the warp-level work-node, we're done!
                        if(!work_left) {
                            break;
                        }
                        /// Now have all the warp-controllers communicate their work //////////////
                        my_group.parallel_for_work_item( [&] (cl::sycl::h_item<1> local_item) {
                            // If I'm in charge of my work
                            if( warp_level_work_node_acc[warp_id(local_item)] == warp_local_id(local_item) ) {
                                // My work is going to be performed, so I won't need control again
                                my_work_left(local_item) = 0;
                                // Put my first and left-edge into local memory
                                warp_level_first_edge_acc[warp_id(local_item)] = my_first_edge(local_item);
                                warp_level_offset_acc[warp_id(local_item)] = 0;
                                warp_level_last_edge_acc[warp_id(local_item)] = my_last_edge(local_item);
                            }
                        });
                        ///////////////////////////////////////////////////////////////////////////
                        //
                        /// Now have each warp complete its work //////////////////////////////////
                        while(work_left) {
                            work_left = false;
                            // TODO
                        }
                        ///////////////////////////////////////////////////////////////////////////
                    }
                    /////////////////////////////////////////////////////////////////////////////////////////
                 });
             });
             cl_ulong submit_time = bfs_iter_event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                      start_time  = bfs_iter_event.get_profiling_info<cl::sycl::info::event_profiling::command_start>(),
                      end_time    = bfs_iter_event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
             fprintf(stderr, "BFS Submit To Start = %0.2f ms\nBFS Start To End = %0.2f ms\n", (start_time - submit_time)/1000000.0, (end_time - start_time)/1000000.0);
          } catch (cl::sycl::exception const& e) {
              std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
            if(e.get_cl_code() != CL_SUCCESS) {
                std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
            }
              std::exit(1);
          }

          // Wait for iter to finish and handle asynchronous erros as in
          // https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
          try {
            queue.wait_and_throw();
          } catch (cl::sycl::exception const& e) {
            std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
            if(e.get_cl_code() != CL_SUCCESS) {
                std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
            }
            std::exit(1);
          }

          // increment level and swap in/out lists
          level++;
          std::swap(in_worklist_buf, out_worklist_buf);
          auto out_worklist_size_acc = out_worklist_size_buf.get_access<cl::sycl::access::mode::read_write>();
          in_worklist_size = out_worklist_size_acc[0];
          out_worklist_size_acc[0] = 0;
      }
      in_worklist_buf = NULL;
      out_worklist_buf = NULL;
   }  // End SYCL scope
*/
