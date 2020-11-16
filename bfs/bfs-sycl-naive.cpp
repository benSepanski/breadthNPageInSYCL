#include <iostream>

#include <CL/sycl.hpp>

// From libsyclutils
//
// Host_CSR_Graph index_type
#include "host_csr_graph.h"
// SYCL_CSR_Graph
#include "sycl_csr_graph.h"

// node data type
typedef uint64_t node_data_type;
#define INF UINT64_MAX 

class bfs_init;

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

   // Build our queue and report device
   cl::sycl::queue queue(device_selector);
   std::cout << "Running on "
             << queue.get_device().get_info<cl::sycl::info::device::name>()
             << "\n";

   // SYCL Scope
   {
      // Build our sycl graph inside scope so that buffers can be destroyed
      // by destructor
      SYCL_CSR_Graph<node_data_type> sycl_graph(&host_graph);

      // Submit a command group that sets node data to INF or 0 (if it's src node)
      queue.submit([&] (cl::sycl::handler& cgh) {
         auto node_data_acc = sycl_graph.node_data.get_access<cl::sycl::access::mode::discard_write>(cgh);

         cgh.parallel_for<class bfs_init>(cl::sycl::range<1>{sycl_graph.nnodes},
             [=] (cl::sycl::id<1> index) {
                node_data_acc[index] = (index.get(0) == src_node) ? 0 : INF;
             });
      });
   }  // End SYCL scope

   // Verify that node data is all ones
   for(int i = 0; i < host_graph.nnodes; ++i) {
       if( host_graph.node_data[i] != INF) {
           printf("Node %d has data %d != INF\n", i, host_graph.node_data[i]);
       }
   }
                
   return 0;
}
