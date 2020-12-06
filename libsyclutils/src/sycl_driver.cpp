/**
 * sycl_driver.cpp
 *
 * Copied and modified from
 * https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/libgpu/src/skelapp/skel.cu
 *
 * Driver for SYCL apps implemented in this project.
 *
 * Each application must implement
 *  - sycl_main
 *  - output
 *
 *  And may implement
 *  - process_prog_opt  (process options e.g. foo -a <arg>)
 *  - process_prog_arg  (process non-option arguments e.g. foo <arg>)
 *
 *  Look at the bfs/ directory for examples of how to implement these
*/ 
#include <chrono>
#include <unistd.h>

// libsyclutils/include
//
// Host_CSR_Graph
#include "host_csr_graph.h"
// SYCL_CSR_Graph
#include "sycl_csr_graph.h"
// NVIDIA_Selector
#include "nvidia_selector.h"

// Application-implemented functions
extern int sycl_main(SYCL_CSR_Graph&, cl::sycl::queue&);
extern void output(Host_CSR_Graph&, const char *output_file);

int QUIET = 0;
char *INPUT, *OUTPUT;

int CUDA_DEVICE = -1;
size_t num_work_groups = 4;

//mgpu::ContextPtr mgc;

extern const char *prog_opts;
extern const char *prog_usage;
extern const char *prog_args_usage;
extern int process_prog_opt(char optchar, char *optarg);
extern int process_prog_arg(int argc, char *argv[], int arg_start);


int load_graph_and_run_kernel(char *graph_file, cl::sycl::device_selector &dev_sel) {
     // read in graph
     Host_CSR_Graph host_graph;
     host_graph.readFromGR(graph_file);
     // Make sure the graph doesn't have more than 32 bits of nodes
     if(host_graph.nnodes >= std::numeric_limits<uint32_t>::max()) {
         printf("SYCL targeting ptx (NVIDIA) does not support 64-bit atomics. num nodes must be < uint32_max");
         std::exit(1);
     }
   
     // Build an exception handler for the command queue as in
     // https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide/error-handling
    auto exception_handler = [] (cl::sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(cl::sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
                if(e.get_cl_code() != CL_SUCCESS) {
                std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
                }
                std::exit(1);
            }
        }
    };
// Begin SYCL scope
    // return value declared outside of SYCL scope
    int r;
    // Begin SYCL scope
    {
        // Build command queue with profiling enabled and report the device
        cl::sycl::property::queue::enable_profiling enab_prof;
        cl::sycl::property_list prop_list(enab_prof);
        cl::sycl::queue queue(dev_sel, exception_handler, prop_list);
        fprintf(stderr, "Running on %s\n",
                queue.get_device().get_info<cl::sycl::info::device::name>().c_str());


        // Create SYCL graph
        SYCL_CSR_Graph sycl_graph(&host_graph);

        // Explicitly copy graph onto device
        try{
            queue.submit([&] (cl::sycl::handler &cgh) {
                auto row_start_host = sycl_graph.row_start.get_access<
                                        cl::sycl::access::mode::read>(cgh);
                auto row_start_dev = sycl_graph.row_start.get_access<
                                        cl::sycl::access::mode::read_write,
                                        cl::sycl::access::target::global_buffer>(cgh);
                cgh.copy(row_start_host, row_start_dev);
            });
            queue.submit([&] (cl::sycl::handler &cgh) {
                auto edge_dst_host = sycl_graph.edge_dst.get_access<
                                        cl::sycl::access::mode::read>(cgh);
                auto edge_dst_dev = sycl_graph.edge_dst.get_access<
                                        cl::sycl::access::mode::read_write,
                                        cl::sycl::access::target::global_buffer>(cgh);
                cgh.copy(edge_dst_host, edge_dst_dev);
            });
        } catch(cl::sycl::exception const& e) {
            std::cerr << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
            if(e.get_cl_code() != CL_SUCCESS) {
            std::cerr << "OpenCL error code " << e.get_cl_code() << std::endl;
            }
            std::exit(1);
        }
        // wait for copy to finish, throwing asynchronous exception to
        // handler if one is found
        queue.wait_and_throw();
        std::cerr << "Graph copied onto device" << std::endl;
 
        // Run application
        auto startTime = std::chrono::high_resolution_clock::now();
        r = sycl_main(sycl_graph, queue);
        auto endTime = std::chrono::high_resolution_clock::now();
 
        // Report time
        double time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
        fprintf(stderr, "Total time: %u ms\n", (uint64_t) time_in_ms);
        fprintf(stderr, "Total time: %u ns\n", (uint64_t) time_in_ns);
    } // end sycl scope
  
   // Finish
   if(!QUIET)
     output(host_graph, OUTPUT);
 
   return r;
}
 
void usage(int argc, char *argv[]) 
{
  if(strlen(prog_usage)) 
    fprintf(stderr, "usage: %s [-q quiet] [-g gpunum] [-b numblocks] [-o output-file] %s graph-file \n %s\n", argv[0], prog_usage, prog_args_usage);
  else
    fprintf(stderr, "usage: %s [-q quiet] [-g gpunum] [-b numblocks] [-o output-file] graph-file %s\n", argv[0], prog_args_usage);
}

void parse_args(int argc, char *argv[]) 
{
  int c;
  const char *skel_opts = "g:qo:b:";
  char *opts;
  int len = 0;
  
  len = strlen(skel_opts) + strlen(prog_opts) + 1;
  opts = (char *) calloc(1, len);
  strcat(strcat(opts, skel_opts), prog_opts);

  while((c = getopt(argc, argv, opts)) != -1) {
    switch(c) 
      {
      case 'q':
        QUIET = 1;
        break;
      case 'o':
        OUTPUT = optarg; //TODO: copy?
        break;
      case 'g':
        char *end;
        errno = 0;
        CUDA_DEVICE = strtol(optarg, &end, 10);
        if(errno != 0 || *end != '\0') {
          fprintf(stderr, "Invalid GPU device '%s'. An integer must be specified.\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case 'b':
        char *wg_end;
        errno = 0;
        num_work_groups = strtol(optarg, &wg_end, 10);
        if(errno != 0 || *wg_end != '\0') {
          fprintf(stderr, "Invalid number of work-groups '%s'. An integer must be specified.\n", optarg);
          exit(EXIT_FAILURE);
        }
        break;
      case '?':
        usage(argc, argv);
        exit(EXIT_FAILURE);
      default:
        process_prog_opt(c, optarg);
        break;
    }
  }

  if(optind < argc) {
    INPUT = argv[optind];
    if(!process_prog_arg(argc, argv, optind + 1)) {
      usage(argc, argv);
      exit(EXIT_FAILURE);
    }
  }
  else {
    usage(argc, argv);
    exit(EXIT_FAILURE);      
  }
}

/**
 * Parse arguments, load graph and run kernel
 *
 * Uses default selector, or chosen nvidia device
 */
int main(int argc, char *argv[]) {
  if(argc == 1) {
    usage(argc, argv);
    exit(1);
  }

  parse_args(argc, argv);
  
  int r;
  if( CUDA_DEVICE < 0 ) {
      cl::sycl::default_selector dev_selector;
      r = load_graph_and_run_kernel(INPUT, dev_selector);
  }
  else {
      NVIDIA_Selector dev_selector( CUDA_DEVICE );
      r = load_graph_and_run_kernel(INPUT, dev_selector);
  }

  return r;
}
