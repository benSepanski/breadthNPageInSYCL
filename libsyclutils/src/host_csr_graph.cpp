// Host_CSR_Graph index_type node_data_type
#include "host_csr_graph.h"

Host_CSR_Graph::Host_CSR_Graph() {
    nnodes = 0;
    nedges = 0;
    row_start = NULL;
    edge_dst = NULL;
    node_data = NULL;
}

unsigned Host_CSR_Graph::readFromGR(char file[]) {
  // Copied from https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/libgpu/src/csr_graph.cu#L176
  std::ifstream cfile;
  cfile.open(file);

  int masterFD = open(file, O_RDONLY);
  if (masterFD == -1) {
    printf("Host_CSR_Graph::readFromGR: unable to open %s.\n", file);
    return 1;
  }

  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    printf("Host_CSR_Graph::readFromGR: unable to stat %s.\n", file);
    abort();
  }
  size_t masterLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;

  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    printf("Host_CSR_Graph::readFromGR: mmap failed.\n");
    abort();
  }

  auto startTime = std::chrono::system_clock::now();

  // parse file
  uint64_t* fptr                           = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes   = le64toh(*fptr++);
  uint64_t numEdges   = le64toh(*fptr++);
  uint64_t* outIdx    = fptr;
  fptr += numNodes;
  uint32_t* fptr32 = (uint32_t*)fptr;
  uint32_t* outs   = fptr32;
  fptr32 += numEdges;
  if (numEdges % 2)
    fptr32 += 1;

  // cuda.
  this->nnodes = numNodes;
  this->nedges = numEdges;

  printf("nnodes=%d, nedges=%d, sizeEdge=%d.\n", this->nnodes, this->nedges, sizeEdgeTy);
  this->allocSpace();

  row_start[0] = 0;

  for (unsigned ii = 0; ii < this->nnodes; ++ii) {
    row_start[ii + 1] = le64toh(outIdx[ii]);
    //   //noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
    index_type degree = this->row_start[ii + 1] - this->row_start[ii];

    for (unsigned jj = 0; jj < degree; ++jj) {
      unsigned edgeindex = this->row_start[ii] + jj;

      unsigned dst = le32toh(outs[edgeindex]);
      if (dst >= this->nnodes)
        printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj,
               edgeindex);

      this->edge_dst[edgeindex] = dst;
    }

    progressPrint(this->nnodes, ii);
  }

  cfile.close(); // probably galois doesn't close its file due to mmap.
  auto endTime = std::chrono::system_clock::now();
  double time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

  // TODO: fix MB/s
  printf("read %lld bytes in %0.2f ms (%0.2f MB/s)\n\r\n",
         masterLength, time_in_ms, (masterLength / 1000.0) / time_in_ms);

  return 0;
}

// Copied from
// https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/libgpu/src/csr_graph.cu#L28
unsigned Host_CSR_Graph::allocSpace() {
  assert(this->nnodes > 0);

  if (this->row_start != NULL) // already allocated
    return true;

  size_t mem_usage = ((this->nnodes + 1) + this->nedges) * sizeof(index_type) +
                     (this->nnodes) * sizeof(node_data_type);

  printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);

  this->row_start = (index_type*)calloc(this->nnodes + 1, sizeof(index_type));
  this->edge_dst  = (index_type*)calloc(this->nedges, sizeof(index_type));
  this->node_data = (node_data_type*)calloc(this->nnodes, sizeof(node_data_type));

  return (this->row_start && this->edge_dst && this->node_data);
}

// Copied from https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/libgpu/src/csr_graph.cu#L156
void Host_CSR_Graph::progressPrint(unsigned maxii, unsigned ii) {
  const unsigned nsteps = 10;
  unsigned ineachstep   = (maxii / nsteps);
  if (ineachstep == 0)
    ineachstep = 1;
  /*if (ii == maxii) {
    printf("\t100%%\n");
    } else*/
  if (ii % ineachstep == 0) {
    int progress = ((size_t)ii * 100) / maxii + 1;

    printf("\t%3d%%\r", progress);
    fflush(stdout);
  }
}
