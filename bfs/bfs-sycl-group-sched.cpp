#include <chrono>
#include <iostream>
#include <limits>
#include <CL/cl.h>
#include <CL/sycl.hpp>

// From libsyclutils
//
// Host_CSR_Graph index_type node_data_type
#include "host_csr_graph.h"
// SYCL_CSR_Graph
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
#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32

// classes used to name SYCL kernels
class node_level_inf;
class init_start_node;
class bfs_iter;


/**
 * Run BFS on the sycl_graph from start_node, storing each node's level
 * into the node_data
 */
void sycl_bfs(SYCL_CSR_Graph &sycl_graph, sycl::queue &queue) {
    /// Put some constants on a buffer and initialize to avoid host-device loops ////////////////////////////
    //
    const index_type NNODES = sycl_graph.nnodes,
                     START_NODE = start_node;
    sycl::buffer<index_type, 1> NNODES_buf(&NNODES, sycl::range<1>{1}),
                                START_NODE_buf(&START_NODE, sycl::range<1>{1}),
                                INF_buf(&INF, sycl::range<1>{1});
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /// Initialize the node data and worklists //////////////////////////////////////////////////////////////
    //
    // Make alternating "in"-worklists and "out"-worklists
    sycl::buffer<index_type, 1> worklist1_buf(sycl::range<1>{NNODES}),
                                worklist2_buf(sycl::range<1>{NNODES}),
                                *in_worklist_buf = &worklist1_buf,
                                *out_worklist_buf = &worklist2_buf;
    // Make buffers to hold the worklist sizes
    gpu_size_t in_worklist_size;
    sycl::buffer<gpu_size_t, 1> in_worklist_size_buf(&in_worklist_size, sycl::range<1>{1}),
                                out_worklist_size_buf(sycl::range<1>{1});
    // set all levels to INF
    queue.submit([&] (sycl::handler &cgh) {
        // get access to node level
        auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::discard_write>(cgh);
        // get INF on the device
        auto INF_acc = INF_buf.get_access<sycl::access::mode::read,
                                          sycl::access::target::constant_buffer>(cgh);
        // Set all node levels to INF
        cgh.parallel_for<class node_level_inf>(sycl::range<1>{NNODES}, [=](sycl::item<1> my_item) {
            node_level[my_item] = INF_acc[0];
        });
    });
    // Set source node level to 0 and initialize worklists
    queue.submit([&] (sycl::handler &cgh) {
        // write to node level
        auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::write>(cgh);
        // dicard-write to in_worklist
        auto in_worklist = in_worklist_buf->get_access<sycl::access::mode::discard_write>(cgh);
        // need access to start node and worklist sizes
        auto START_NODE_acc = START_NODE_buf.get_access<sycl::access::mode::read,
                                                        sycl::access::target::constant_buffer>(cgh);
        auto in_worklist_size = in_worklist_size_buf.get_access<sycl::access::mode::discard_write>(cgh);
        auto out_worklist_size = out_worklist_size_buf.get_access<sycl::access::mode::discard_write>(cgh);
        // set source node level to 0 and initialize worklists
        cgh.single_task<class init_start_node>([=]() {
            node_level[START_NODE_acc[0]] = 0;
            in_worklist[0] = START_NODE_acc[0];
            in_worklist_size[0] = 1;
            out_worklist_size[0] = 0;
        });
    });
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Get work_group_size onto device to avoid copies
    const size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
    sycl::buffer<const size_t, 1> WORK_GROUP_SIZE_buf(&WORK_GROUP_SIZE, sycl::range<1>{1});
    /// Now invoke BFS routine at each level ////////////////////////////////////////////////////////////////
    size_t level = 1, level_host_copy = 1, in_worklist_size_host_copy = 1;
    sycl::buffer<size_t, 1> LEVEL_buf(&level, sycl::range<1>{1});
    do {
        /// Run an iteration of BFS ///////////////////////////////////////////////////////////////
        queue.submit([&] (sycl::handler &cgh) {
            // constants for work distribution
            const size_t NUM_WORK_GROUPS = (in_worklist_size_host_copy + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

            // get constant access to work-group size on device
            auto WORK_GROUP_SIZE_acc = WORK_GROUP_SIZE_buf.get_access<sycl::access::mode::read,
                                                                      sycl::access::target::constant_buffer>(cgh);
            // Get read-access to the CSR graph
            auto row_start = sycl_graph.row_start.get_access<sycl::access::mode::read>(cgh);
            auto edge_dst  = sycl_graph.edge_dst.get_access<sycl::access::mode::read>(cgh);
            // Get read-write access to the node's level
            auto node_level = sycl_graph.node_data.get_access<sycl::access::mode::read_write>(cgh);
            // Get constant access to the number of nodes and the INF constant
            auto NNODES_acc = NNODES_buf.get_access<sycl::access::mode::read,
                                                    sycl::access::target::constant_buffer>(cgh);
            auto INF_acc = INF_buf.get_access<sycl::access::mode::read,
                                              sycl::access::target::constant_buffer>(cgh);
            // We need to read from the worklist and its size
            auto in_worklist = in_worklist_buf->get_access<sycl::access::mode::read>(cgh);
            auto IN_WORKLIST_SIZE = in_worklist_size_buf.get_access<sycl::access::mode::read,
                                                                    sycl::access::target::constant_buffer>(cgh);
            // We need to (discard) write to the out-worklist and its size
            auto out_worklist = out_worklist_buf->get_access<sycl::access::mode::discard_write>(cgh);
            auto out_worklist_size = out_worklist_size_buf.get_access<sycl::access::mode::atomic>(cgh);

            /// Now submit our bfs job //////////////////////////////////////////////////
            cgh.parallel_for_work_group<class bfs_iter>(sycl::range<1>{NUM_WORK_GROUPS}, sycl::range<1>{WORK_GROUP_SIZE},
            [=] (sycl::group<1> my_group) {
                // Have each node in my group figure out what work it needs to do (if any)
                sycl::private_memory<index_type> my_node(my_group),
                                                 my_first_edge(my_group),
                                                 my_last_edge(my_group),
                                                 my_work_left(my_group);
                my_group.parallel_for_work_item([&] (sycl::h_item<1> my_item) {
                    sycl::id<1> global_id = my_item.get_global_id();
                    if(global_id[0] < IN_WORKLIST_SIZE[0]) {
                        my_node(my_item) = in_worklist[global_id];
                        my_first_edge(my_item) = row_start[my_node(my_item)];
                        my_last_edge(my_item) = row_start[my_node(my_item)+1];
                    }
                    else {
                        my_node(my_item) = INF;
                        my_first_edge(my_item) = INF;
                        my_last_edge(my_item) = INF;
                    }
                    my_work_left(my_item) = my_last_edge(my_item) - my_first_edge(my_item);
                });
                /// group-scheduling //////////////////////////////////////////
                //(work as group til no one wants group control)
                index_type prev_work_node, work_node = WORK_GROUP_SIZE_acc[0];
                while(true) {
                    prev_work_node = work_node;
                    // If no-one competed for control of the group, we're done!
                    if(prev_work_node == work_node) {
                        break;
                    }
                }
                ///////////////////////////////////////////////////////////////
            });
            /////////////////////////////////////////////////////////////////////////////
        });
        ///////////////////////////////////////////////////////////////////////////////////////////

        /// Get ready for next iteration //////////////////////////////////////////////////////////
        //
        // Wait for iteration to finish and throw asynchronous errors if any
        queue.wait_and_throw();
        // Increment level (avoiding a copy device->host)
        auto level_acc = LEVEL_buf.get_access<sycl::access::mode::discard_write>();
        level_acc[0] = level_host_copy + 1;
        level_host_copy++;
        // Set new in-worklist size and reset out-worklist size to zero
        auto out_worklist_size_acc = out_worklist_size_buf.get_access<sycl::access::mode::read_write>();
        in_worklist_size_host_copy = out_worklist_size_acc[0];
        out_worklist_size_acc[0] = 0;
        auto in_worklist_size_acc = in_worklist_size_buf.get_access<sycl::access::mode::discard_write>();
        in_worklist_size_acc[0] = in_worklist_size_host_copy;
        // Swap worklists
        std::swap(in_worklist_buf, out_worklist_buf);
        ///////////////////////////////////////////////////////////////////////////////////////////
    } while(level_host_copy < INF && in_worklist_size_host_copy > 0);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
}


int sycl_main(Host_CSR_Graph &host_graph, sycl::queue &queue) {
   // copy start_node into local variable so we can use it inside SYCL kernels
   const index_type START_NODE = start_node;

   // SYCL Scope
   {
      // Build our sycl graph inside scope so that buffers can be destroyed
      // by destructor
      SYCL_CSR_Graph sycl_graph(&host_graph);

      try {
          sycl_bfs(sycl_graph, queue);
      } catch (cl::sycl::exception const& e) {
          std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
          if(e.get_cl_code() != CL_SUCCESS) {
              std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
          }
          std::exit(1);
      }
   } // End SYCL Scope

   return 0;
}
   /**
    *

          // Set node data to INF or 0 (if it's src node)
          cl::sycl::event bfs_init_event = queue.submit([&] (cl::sycl::handler& cgh) {
             auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::discard_write>(cgh);

             cgh.parallel_for<class bfs_init>(cl::sycl::range<1>{sycl_graph.nnodes},
                 [=] (cl::sycl::id<1> index) {
                    node_data_acc[index] = (index.get(0) == START_NODE) ? 0 : INF;
                 });
          });
          cl_ulong submit_time = bfs_init_event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>(),
                   start_time  = bfs_init_event.get_profiling_info<cl::sycl::info::event_profiling::command_start>(),
                   end_time    = bfs_init_event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
          fprintf(stderr, "BFS_Init Submit To Start = %0.2f ms\nBFS_Init Start To End = %0.2f ms\n", (start_time - submit_time)/1000000.0, (end_time - start_time)/1000000.0);
      } catch (cl::sycl::exception const& e) {
          std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
          std::exit(1);
      }

      // Build a worklist buffer of nodes
      uint32_t in_worklist_size = 1, out_worklist_size = 0;
      cl::sycl::buffer<index_type, 1> buf1(cl::sycl::range<1>{sycl_graph.nnodes}),
                                      buf2(cl::sycl::range<1>{sycl_graph.nnodes}),
                                      *in_worklist_buf = &buf1,
                                      *out_worklist_buf = &buf2;
      cl::sycl::buffer<uint32_t, 1> out_worklist_size_buf(&out_worklist_size, cl::sycl::range<1>{1});

      // initialize src node on in worklist
      auto in_wl_acc = in_worklist_buf->get_access<cl::sycl::access::mode::discard_write>();
      in_wl_acc[0] = START_NODE;
 
      // Run bfs for each level
      node_data_type level = 1;
      while( in_worklist_size > 0 && level < INF) {
          // run an iteration of bfs at the given level
          try {
              cl::sycl::event bfs_iter_event = queue.submit([&] (cl::sycl::handler &cgh) {
              cl::sycl::stream sycl_stream(1024, 256, cgh);
              // save current level in command group
              const node_data_type LEVEL = level;

              // Determine number of groups
              const uint32_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
              uint32_t NUM_WORK_GROUPS = (in_worklist_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
              uint32_t WARPS_PER_GROUP = (WORK_GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE;
              const uint32_t GROUP_LOCAL_QUEUE_SIZE = 16 * THREAD_BLOCK_SIZE;

              // Some tuning constants
              // must have degree >= MIN_GROUP_SCHED_DEGREE to do thread-block scheduling
              //const uint32_t MIN_GROUP_SCHED_DEGREE = WORK_GROUP_SIZE + 1;
              const uint32_t MIN_GROUP_SCHED_DEGREE = 1;

              // get accessors
              auto row_start_acc = sycl_graph.row_start.get_access<cl::sycl::access::mode::read>(cgh);
              auto edge_dst_acc = sycl_graph.edge_dst.get_access<cl::sycl::access::mode::read>(cgh);
              auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::read_write>(cgh);

              auto in_worklist_acc = in_worklist_buf->get_access<cl::sycl::access::mode::read>(cgh);
              auto out_worklist_acc = out_worklist_buf->get_access<cl::sycl::access::mode::discard_write>(cgh);

              auto out_worklist_size_acc = out_worklist_size_buf.get_access<cl::sycl::access::mode::atomic>(cgh);

              // Give each group a group-local queue
              cl::sycl::accessor<index_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                    group_local_queue_acc(cl::sycl::range<1>{GROUP_LOCAL_QUEUE_SIZE}, cgh);
              // Give each group a space to perform warp-level scheduling
              cl::sycl::accessor<index_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                    warp_level_work_node_acc(cl::sycl::range<1>{WARPS_PER_GROUP}, cgh),
                    warp_level_first_edge_acc(cl::sycl::range<1>{WARPS_PER_GROUP}, cgh),
                    warp_level_last_edge_acc(cl::sycl::range<1>{WARPS_PER_GROUP}, cgh),
                    warp_level_offset_acc(cl::sycl::range<1>{WARPS_PER_GROUP}, cgh);

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

  return 0;
}
*/
