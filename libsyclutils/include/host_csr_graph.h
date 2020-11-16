/**
 * host_csr_graph.h
 *
 * A simple CSR Graph based on the implementation in
 * https://github.com/IntelligentSoftwareSystems/Galois/blob/306535c4931b8d398518624b9b6428f7120a0b44/libgpu/include/csr_graph.h#L1
 * 
 * but able to be compiled into a SYCL project
 */
#ifndef BREADTHNPAGEINSYCL_SYCLUTILS_HOST_CSR_GRAPH_
#define BREADTHNPAGEINSYCL_SYCLUTILS_HOST_CSR_GRAPH_

#include <cassert>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>


typedef size_t index_type ;

/**
 * A very simple CSR graph:
 *      - Can read from a *.gr file
 *
 *  Has node data but no edge data
 */ 
template <typename node_data_type>
struct Host_CSR_Graph {
    // num nodes, num edges, index of first edge, edge destinations
    index_type nnodes, nedges, *row_start, *edge_dst;
    // node data
    node_data_type *node_data;

    /** Create an uninitialized CSR graph */
    Host_CSR_Graph() ;  

    /**
     * read a graph from a *.gr file into this object
     *
     * This function will read a directed graph from a *.gr
     * file into this object.
     * It will ignore edge data.
     *
     * @param file The name of the *.gr file to read
     * @param 
     */
    unsigned readFromGR(char file[]);

    /**
     * @param node the index of a node
     * @return true iff *node* is a valid index
     */
    bool is_valid_node(index_type node) const {
        return (node < this->nnodes);
    }

    /**
     * @param edge the index of an edge 
     * @return true iff *edge* is a valid index
     */
    bool is_valid_edge(index_type edge) const {
        return (edge < this->nedges);
    }

    /**
     * Get the out-degree of a node. Asserts that the node is valid
     *
     * @param node the index of a node
     * @return the degree of that node
     */
    index_type get_out_degree(index_type node) const {
        assert( this->is_valid_node(node) );

        return this->row_start[node];
    }

    /*
     * Get the specified neighbor of node
     *
     * Asserts that *node* is a valid node and that
     * *local_edge_nr < degree(node)*
     *
     * @param node the source node
     * @param local_edge_nr which out-edge from node to follow
     *
     * @return the destination of the *local_edge_nr*th edge of *node*
     */
    index_type get_neighbor(index_type node, index_type local_edge_nr) const {
        assert( this->is_valid_node(node) );
        assert( local_edge_nr < this->get_out_degree(node) );

        index_type edge_index = this->row_start[node] + local_edge_nr;

        assert( this->is_valid_edge(edge_index) );
        return this->edge_dst[edge_index];
    }

    /*
     * Get destination of the edge
     *
     * Asserts that *edge* is a valid edge index
     *
     * @param edge the index of an edge
     *
     * @return the destination of *edge)
     */
    index_type get_neighbor(index_type edge) const {
        assert( this->is_valid_edge(edge) );
        return this->edge_dst[edge];
    }

    /** 
     * Get first edge of a node
     *
     * asserts the *node* is valid
     *
     * @param node the index of the node to get the first edge from
     * @return the index of the first edge of *node*
     */
    index_type get_first_edge(index_type node) const {
        assert( this->is_valid_node(node) );
        return this->row_start[node];
    }

    /** 
     * Get data at a node
     *
     * asserts the *node* is valid
     *
     * @param node the index of the node to get data from
     * @return the data at *node*
     */
    node_data_type get_node_data(index_type node) const {
        assert( this->is_valid_node(node) );
        return this->node_data[node];
    }

    private:
        /** allocate the arrays in memory 
         * 
         * @return true if successful
         * */
        unsigned allocSpace() ;  

        /** Print utility used by `readFromGR` */
        void progressPrint(unsigned maxii, unsigned ii);
};

////// End of declaration


////// Definitions

template <typename node_data_type>
Host_CSR_Graph<node_data_type>::Host_CSR_Graph() {
    nnodes = 0;
    nedges = 0;
    row_start = NULL;
    edge_dst = NULL;
    node_data = NULL;
}

template <typename node_data_type>
unsigned Host_CSR_Graph<node_data_type>::readFromGR(char file[]) {
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

  std::clock_t start = std::clock();

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

      progressPrint(this->nnodes, ii);
    }
  }

  cfile.close(); // probably galois doesn't close its file due to mmap.
  double time = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

  // TODO: fix MB/s
  printf("read %lld bytes in %d ms (%0.2f MB/s)\n\r\n", masterLength,
         time * 1000, (masterLength / 1000.0) / (time * 1000));

  return 0;
}

// Copied from
// https://github.com/IntelligentSoftwareSystems/Galois/blob/c6ab08b14b1daa20d6b408720696c8a36ffe30cb/libgpu/src/csr_graph.cu#L28
template <typename node_data_type>
unsigned Host_CSR_Graph<node_data_type>::allocSpace() {
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
template <typename node_data_type>
void Host_CSR_Graph<node_data_type>::progressPrint(unsigned maxii, unsigned ii) {
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

#endif
