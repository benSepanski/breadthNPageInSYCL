/**
 * sycl_csr_graph.h
 *
 * Just a container for SYCL buffers for the arrays of
 * a CSR graph with node data
 */
#ifndef BREADTHNPAGEINSYCL_SYCLUTILS_SYCL_CSR_GRAPH_
#define BREADTHNPAGEINSYCL_SYCLUTILS_SYCL_CSR_GRAPH_

#include <CL/sycl.hpp>

// from Galois::gpu
//
// HOST_CSR_Graph index_type node_data_type
#include "host_csr_graph.h"

/**
 * A CSR graph with node data represented
 * as SYCL buffers
 */
struct SYCL_CSR_Graph {
    index_type nnodes, nedges;
    // All are 1-D buffers
    cl::sycl::buffer<index_type, 1> row_start, edge_dst;
    cl::sycl::buffer<node_data_type, 1> node_data;

    /** Construct SYCL_CSR_Graph from a CSR_Graph */
    SYCL_CSR_Graph( Host_CSR_Graph *graph )
        : nnodes   {graph->nnodes}
        , nedges   {graph->nedges}
        , row_start{graph->row_start, cl::sycl::range<1>{graph->nnodes+1}}
        , edge_dst {graph->edge_dst,  cl::sycl::range<1>{graph->nedges}}
        , node_data{graph->node_data, cl::sycl::range<1>{graph->nnodes}}
        { }
};

#endif
