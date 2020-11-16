#include <iostream>
#include <limits>
#include <CL/sycl.hpp>

// From libsyclutils
//
// Host_CSR_Graph index_type
#include "host_csr_graph.h"
// SYCL_CSR_Graph
#include "sycl_csr_graph.h"

// from support.cpp
extern void output(const Host_CSR_Graph<uint64_t>&, char[]);

typedef uint64_t node_data_type;
// defined here, but const so need to declare as extern so support.cpp
// can use it
extern const uint64_t INF = std::numeric_limits<uint64_t>::max();

// classes used to name SYCL kernels
class bfs_init;
class bfs_iter;

int main(int, char**) {
   // TODO: Take selector, graph, src node as cmd line arguments
    
   // Read in graph
   char file[] = "/net/ohm/export/iss/dist-inputs/rmat15.gr";
   Host_CSR_Graph<node_data_type> host_graph;
   host_graph.readFromGR(file);

   // Assume source is 0
   index_type src_node = 0;

   // Select default device
   cl::sycl::default_selector device_selector;

   // From https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
   // to catch asynchronous exceptions
   auto exception_handler = [] (cl::sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch(cl::sycl::exception const& e) {
          std::cout << "Caught asynchronous SYCL exception:\n"
                    << e.what() << std::endl;
        }
      }
    };

   // Build our queue and report device
   cl::sycl::queue queue(device_selector, exception_handler);
   std::cout << "Running on "
             << queue.get_device().get_info<cl::sycl::info::device::name>()
             << "\n";

   // SYCL Scope
   {
      // Build our sycl graph inside scope so that buffers can be destroyed
      // by destructor
      SYCL_CSR_Graph<node_data_type> sycl_graph(&host_graph);

      try {
      // Set node data to INF or 0 (if it's src node)
      queue.submit([&] (cl::sycl::handler& cgh) {
         auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::discard_write>(cgh);

         cgh.parallel_for<class bfs_init>(cl::sycl::range<1>{sycl_graph.nnodes},
             [=] (cl::sycl::id<1> index) {
                node_data_acc[index] = (index.get(0) == src_node) ? 0 : INF;
             });
      });
      } catch (cl::sycl::exception const& e) {
      std::cout << "Caught synchronous SYCL exception:\n"
                << e.what() << std::endl;
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
      in_wl_acc[0] = src_node;
 
      // Run bfs for each level
      node_data_type level = 1;
      while( in_worklist_size > 0 && level < INF) {
          printf("Running level %d, worklist size=%d\n", level, in_worklist_size);
          // run an iteration of bfs at the given level
          queue.submit([&] (cl::sycl::handler &cgh) {
              // for i/o
              //cl::sycl::stream sycl_stream(1024, 256, cgh);

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

  // error handling as in https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
  try {
      queue.wait_and_throw();
  } catch (cl::sycl::exception const& e) {
  std::cout << "Caught synchronous SYCL exception:\n"
            << e.what() << std::endl;
  }



   // Write to outputfile
   // TODO : Take output file from command line
   char out_file[] = "rmat15.txt";
   output(host_graph, out_file);

   return 0;
}
