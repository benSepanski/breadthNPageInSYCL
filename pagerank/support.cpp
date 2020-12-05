#include <climits>
#include <cstring>
#include <float.h>

// from libsyclutils
//
// Host_CSR_Graph index_type
#include "host_csr_graph.h"

// from bfs-sycl-naive.cpp
extern const uint64_t INF;

// Copied and modified from
// https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/lonestar/analytics/gpu/bfs/support.cu#L7-L43
struct pr_value {
  index_type node;
  float rank;
  inline bool operator< (const pr_value& rhs) const {
    return rank < rhs.rank;
  }
};

/* TODO: accept ALPHA and EPSILON */
const char *prog_opts = "nt:x:";
const char *prog_usage = "[-n] [-t top_ranks] [-x max_iterations]";
const char *prog_args_usage = "";

extern float *P_CURR, *P_NEXT;
extern const float ALPHA, EPSILON;
extern int MAX_ITERATIONS;
extern int iterations;

int NO_PRINT_PAGERANK = 0;
int PRINT_TOP = 0;
int MAX_ITERATIONS =  INT_MAX;

int process_prog_arg(int argc, char *argv[], int arg_start) {
   return 1;
}

void process_prog_opt(char c, char *optarg) {
  if(c == 'n')
    NO_PRINT_PAGERANK = 1;

  if(c == 't') {
    PRINT_TOP = atoi(optarg);    
  }

  if(c == 'x') {
    MAX_ITERATIONS = atoi(optarg);
  }
}

// Copied and modified from 
// https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/lonestar/analytics/gpu/pagerank/support.cu#L45-L100
// 
// This way we can compare outputs of our file with the lonestar gpu outputfile
// using just a diff
void output(Host_CSR_Graph &g, const char *output_file) {
  FILE *f;

  struct pr_value * pr;

  pr = (struct pr_value *) calloc(g.nnodes, sizeof(struct pr_value));

  if(pr == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    exit(1);
  }

  fprintf(stderr, "PR took %d iterations\n", iterations);
  fprintf(stderr, "Calculating sum ...\n");
  float sum = 0;
  for(int i = 0; i < g.nnodes; i++) {
    pr[i].node = i;
    pr[i].rank = P_CURR[i];
    sum += P_CURR[i];
  }

  fprintf(stdout, "sum: %f (%d)\n", sum, g.nnodes);

  if(!output_file)
    return;

//  fprintf(stderr, "Sorting by rank ...\n");
//  std::stable_sort(pr, pr + g.nnodes);
//  fprintf(stderr, "Writing to file ...\n");

  if(strcmp(output_file, "-") == 0)
    f = stdout;
  else
    f = fopen(output_file, "w");

//  fprintf(f, "ALPHA %*e EPSILON %*e\n", FLT_DIG, ALPHA, FLT_DIG, EPSILON);

  if(PRINT_TOP == 0)
    PRINT_TOP = g.nnodes;

//  fprintf(f, "RANKS 1--%d of %d\n", PRINT_TOP, g.nnodes);

  /* for(int i = 1; i <= PRINT_TOP; i++) {
    if(NO_PRINT_PAGERANK) 
      fprintf(f, "%d %d\n", i, pr[g.nnodes - i].node);
    else 
      fprintf(f, "%d %d %*e\n", i, pr[g.nnodes - i].node, FLT_DIG, pr[g.nnodes - i].rank/sum);  
  } */
  for(int i = 0; i < g.nnodes; i++) {
    if(NO_PRINT_PAGERANK) 
      fprintf(f, "%d\n", pr[i].node);
    else 
      fprintf(f, "%d %f\n", pr[i].node, FLT_DIG, pr[i].rank);  
  }

  free(pr);
}
