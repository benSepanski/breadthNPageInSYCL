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
// NVIDIA_Selector
#include "nvidia_selector.h"

// Application-implemented functions
extern int sycl_main(Host_CSR_Graph&, cl::sycl::device_selector&);
extern void output(Host_CSR_Graph&, const char *output_file);

int QUIET = 0;
char *INPUT, *OUTPUT;

int CUDA_DEVICE = -1;

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

  // Run application
  auto startTime = std::chrono::system_clock::now();
  int r = sycl_main(host_graph, dev_sel);
  auto endTime = std::chrono::system_clock::now();

  // Report time
  double time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  double time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
  fprintf(stderr, "Total time: %u ms\n", (uint64_t) time_in_ms);
  fprintf(stderr, "Total time: %u ns\n", (uint64_t) time_in_ns);

  // Finish
  if(!QUIET)
    output(host_graph, OUTPUT);

  return r;
}

void usage(int argc, char *argv[]) 
{
  if(strlen(prog_usage)) 
    fprintf(stderr, "usage: %s [-q quiet] [-g gpunum] [-o output-file] %s graph-file \n %s\n", argv[0], prog_usage, prog_args_usage);
  else
    fprintf(stderr, "usage: %s [-q quiet] [-g gpunum] [-o output-file] graph-file %s\n", argv[0], prog_args_usage);
}

void parse_args(int argc, char *argv[]) 
{
  int c;
  const char *skel_opts = "g:qo:";
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
