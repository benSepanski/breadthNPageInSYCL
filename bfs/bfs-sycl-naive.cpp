#include <iostream>
#include <limits>
#include <CL/cl.h>
#include <CL/sycl.hpp>

// From libsyclutils
//
// SYCL_CSR_Graph index_type node_data_type
#include "sycl_csr_graph.h"

// from support.cpp
extern index_type start_node;

// defined here, but const so need to declare as extern so support.cpp
// can use it
extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

// classes used to name SYCL kernels
class bfs_init;
class bfs_iter;

int sycl_main(SYCL_CSR_Graph &sycl_graph, cl::sycl::queue &queue) {
   // copy start_node into local variable so we can use it inside SYCL kernels
   const index_type START_NODE = start_node;

   // SYCL Scope
   {
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
              // save current level in command group
              const node_data_type LEVEL = level;

              // get accessors
              auto row_start_acc = sycl_graph.row_start.get_access<cl::sycl::access::mode::read>(cgh);
              auto edge_dst_acc = sycl_graph.edge_dst.get_access<cl::sycl::access::mode::read>(cgh);
              auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::read_write>(cgh);

              auto in_worklist_acc = in_worklist_buf->get_access<cl::sycl::access::mode::read>(cgh);
              auto out_worklist_acc = out_worklist_buf->get_access<cl::sycl::access::mode::discard_write>(cgh);

              auto out_worklist_size_acc = out_worklist_size_buf.get_access<cl::sycl::access::mode::atomic>(cgh);

              // put the bfs iter on the command queue
              cgh.parallel_for<class bfs_iter>(cl::sycl::range<1>{in_worklist_size},
                [=] (cl::sycl::id<1> index) {
                    // get the src_node
                    index_type node = in_worklist_acc[index];
                    // for each neighbor dst_node
                    for(index_type edge_index = row_start_acc[node] ; edge_index < row_start_acc[node+1] ; ++edge_index) {
                        index_type dst_node = edge_dst_acc[edge_index];
                        // Update node data if closer
                        if( node_data_acc[dst_node] == INF ) {
                            node_data_acc[dst_node] = LEVEL ;
                            cl::sycl::atomic<uint32_t> atomic_wl_size = out_worklist_size_acc[0];
                            uint32_t wl_index = atomic_wl_size.fetch_add(1);
                            out_worklist_acc[wl_index] = dst_node;
                        }
                    }
              });
          });
          } catch (cl::sycl::exception const& e) {
              std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
              std::exit(1);
          }

          // wait for iter to finish
          queue.wait();

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

  // asynchronous error handling as in
  // https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
  try {
    queue.wait_and_throw();
  } catch (cl::sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
    std::exit(1);
  }

  return 0;
}
