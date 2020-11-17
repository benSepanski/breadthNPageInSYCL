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
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>


typedef size_t index_type ;
typedef uint64_t node_data_type ;

/**
 * A very simple CSR graph:
 *      - Can read from a *.gr file
 *
 *  Has node data but no edge data
 */ 
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

#endif
