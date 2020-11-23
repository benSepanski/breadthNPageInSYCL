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

// from support.cpp
extern index_type start_node;

typedef uint64_t node_data_type;
// defined here, but const so need to declare as extern so support.cpp
// can use it
extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

// TODO: Make these extern consts which are determined by sycl_driver.cpp
#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32

// classes used to name SYCL kernels
class bfs_init;
class bfs_iter;

int sycl_main(Host_CSR_Graph &host_graph, cl::sycl::device_selector &dev_selector) {
   // From https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
   // to catch asynchronous exceptions
   auto exception_handler = [] (cl::sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch(cl::sycl::exception const& e) {
          std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
          std::exit(1);
        }
      }
    };

   // Build our queue and report device
   cl::sycl::queue queue(dev_selector, exception_handler);
   std::cerr << "Running on "
             << queue.get_device().get_info<cl::sycl::info::device::name>()
             << "\n";

   // copy start_node into local variable so we can use it inside SYCL kernels
   const index_type START_NODE = start_node;

   // SYCL Scope
   {
      // Build our sycl graph inside scope so that buffers can be destroyed
      // by destructor
      SYCL_CSR_Graph sycl_graph(&host_graph);

      try {
          // Set node data to INF or 0 (if it's src node)
          queue.submit([&] (cl::sycl::handler& cgh) {
             auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::discard_write>(cgh);

             cgh.parallel_for<class bfs_init>(cl::sycl::range<1>{sycl_graph.nnodes},
                 [=] (cl::sycl::id<1> index) {
                    node_data_acc[index] = (index.get(0) == START_NODE) ? 0 : INF;
                 });
          });
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
          queue.submit([&] (cl::sycl::handler &cgh) {
              cl::sycl::stream sycl_stream(1024, 256, cgh);
              // save current level in command group
              const node_data_type LEVEL = level;

              // Determine number of groups
              const uint32_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE;
              uint32_t NUM_WORK_GROUPS = (in_worklist_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

              // get accessors
              auto row_start_acc = sycl_graph.row_start.get_access<cl::sycl::access::mode::read>(cgh);
              auto edge_dst_acc = sycl_graph.edge_dst.get_access<cl::sycl::access::mode::read>(cgh);
              auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::read_write>(cgh);

              auto in_worklist_acc = in_worklist_buf->get_access<cl::sycl::access::mode::read>(cgh);
              auto out_worklist_acc = out_worklist_buf->get_access<cl::sycl::access::mode::discard_write>(cgh);

              auto out_worklist_size_acc = out_worklist_size_buf.get_access<cl::sycl::access::mode::atomic>(cgh);

              // Give each group a group-local queue
              cl::sycl::accessor<index_type, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                    group_local_queue_acc(cl::sycl::range<1>{WORK_GROUP_SIZE}, cgh);

              // put the bfs iter on the command queue
              cgh.parallel_for_work_group<class bfs_iter>(cl::sycl::range<1>{NUM_WORK_GROUPS},
                                                          cl::sycl::range<1>{WORK_GROUP_SIZE},
                [=] (cl::sycl::group<1> my_group) {
                    // initialize work_node to an invalid group id
                    size_t work_node = WORK_GROUP_SIZE, prev_work_node;

                    // Have each work-item figure out the node it needs to work on (if any)
                    cl::sycl::private_memory<index_type> my_node(my_group);
                    my_group.parallel_for_work_item(cl::sycl::range<1>{WORK_GROUP_SIZE},
                        [&] (cl::sycl::h_item<1> local_item) {
                            if(local_item.get_global_id()[0] < in_worklist_size) {
                                my_node(local_item) = in_worklist_acc[local_item.get_global_id()];
                            }
                            else {
                                my_node(local_item) = WORK_GROUP_SIZE;
                            }
                        });

                    // Now do work in my group until nobody in my group has work left
                    // to do
                    while(true) {
                        prev_work_node = work_node;
                        // Compete for the work_node
                        my_group.parallel_for_work_item(cl::sycl::range<1>{WORK_GROUP_SIZE},
                            [&](cl::sycl::h_item<1> local_item) {
                                // if I got to be the work-node last time, my_node is complete
                                if(my_node(local_item) == work_node) {
                                    my_node(local_item) = WORK_GROUP_SIZE;
                                }
                                else if(my_node(local_item) != WORK_GROUP_SIZE) {
                                    work_node = my_node(local_item);
                                }
                            });
                        // If nobody claimed the work node, we're done!
                        if(work_node == prev_work_node) break;
                        sycl_stream << "work_node=" << work_node << cl::sycl::endl;

                        // get the first edge and out-degree
                        index_type first_edge = row_start_acc[work_node],
                                   last_edge = row_start_acc[work_node+1];

                        // Now distribute work on the work_node's out edges across this group
                        // in blocks of size WORK_GROUP_SIZE
                        while(first_edge < last_edge) {
                            // empty the group-local queue
                            size_t group_local_queue_size = 0,
                                   group_queue_owner = WORK_GROUP_SIZE,
                                   group_queue_owner_check = WORK_GROUP_SIZE;
                            // get number of edges to work on edges
                            size_t group_size = WORK_GROUP_SIZE;
                            if(group_size > last_edge - first_edge) {
                                group_size = last_edge - first_edge;
                            }
                            sycl_stream << "grp size = " << group_size << cl::sycl::endl;
                            // work on those edges
                            my_group.parallel_for_work_item(cl::sycl::range<1>{group_size}, [&](cl::sycl::h_item<1> local_item) {
                                index_type edge_index = first_edge + local_item.get_local_id()[0];
                                index_type edge_dst = edge_dst_acc[edge_index];
                                // Update node data if closer
                                if( node_data_acc[edge_dst] == INF ) {
                                    node_data_acc[edge_dst] = LEVEL ;
                                    // Compete for ownership of the group-queue
                                    /*
                                    do {
                                        // get pending approval by writing to the check
                                        if(group_queue_owner == WORK_GROUP_SIZE && group_queue_owner_check == WORK_GROUP_SIZE) {
                                            group_queue_owner_check = local_item.get_local_id()[0];
                                        }
                                        // make sure anyone that writes to the check or owner commits their write
                                        // before the upcoming read
                                        my_group.mem_fence(cl::sycl::access::fence_space::local_space);
                                        // If we received pending approval and no-one got approval,
                                        // take full control
                                        if(group_queue_owner == WORK_GROUP_SIZE && group_queue_owner_check == local_item.get_local_id()[0]) {
                                            group_queue_owner = local_item.get_local_id()[0];
                                        }
                                        // make sure these read/writes get committed before the upcoming read
                                        my_group.mem_fence(cl::sycl::access::fence_space::local_space);
                                    } while(group_queue_owner != local_item.get_local_id()[0]);
                                    */
                                    // Now put what you want on the queue
                                    group_local_queue_acc[group_local_queue_size] = edge_dst;
                                    group_local_queue_size++;
                                    // relinquish control of group-local queue
                                    group_queue_owner = WORK_GROUP_SIZE;
                                    group_queue_owner_check = WORK_GROUP_SIZE;
                                }
                            });


                            // Now load the group-local queue onto the queue
                            /*
                            uint32_t offset = out_worklist_size_acc[0].fetch_add(group_local_queue_size);
                            my_group.parallel_for_work_item(cl::sycl::range<1>{group_local_queue_size},
                                [&] (cl::sycl::h_item<1> local_item) {
                                    out_worklist_acc[offset + local_item.get_local_id()[0]] = group_local_queue_acc[local_item.get_local_id()[0]];
                                });
                            */
                            // Now move firstEdge forward
                            first_edge += group_size;
                        }
                     }
                 });
             });
          } catch (cl::sycl::exception const& e) {
              std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
              std::exit(1);
          }

          // Wait for iter to finish and handle asynchronous erros as in
          // https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
          try {
            queue.wait_and_throw();
          } catch (cl::sycl::exception const& e) {
            std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
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
