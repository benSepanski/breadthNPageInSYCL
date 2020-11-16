#include <cstring>
#include <limits>

// from libsyclutils
//
// Host_CSR_Graph
#include "host_csr_graph.h"

// from bfs-sycl-naive.cpp
extern const uint64_t INF;

// Copied from 
// https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/lonestar/analytics/gpu/bfs/support.cu#L27-L47
// 
// This way we can compare outputs of our file with the lonestar gpu outputfile
// using just a diff
void output(const Host_CSR_Graph<uint64_t> &graph, char output_file[]) {
 FILE *f;

  if(!output_file)
    return;

  if(strcmp(output_file, "-") == 0)
    f = stdout;
  else
    f = fopen(output_file, "w");

  const uint32_t infinity = std::numeric_limits<uint32_t>::max() / 4;    
  for(int i = 0; i < graph.nnodes; i++) {
    if(graph.node_data[i] == INF) {
      //formatting the output to be compatible with the distributed bfs ouput 
      fprintf(f, "%d %d\n", i, infinity);
    } else {
      fprintf(f, "%d %d\n", i, graph.node_data[i]);
    }    
  }
}
