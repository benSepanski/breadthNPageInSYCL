/*  -*- mode: c++ -*- */
#include <CL/sycl.hpp>

// index_type
#include "sycl_csr_graph.h"
// Pipe gpu_size_t
#include "pipe.h"

#ifndef BREADTHNPAGEINSYCL_LIBSYCLUTILS_INWORKLIST_
#define BREADTHNPAGEINSYCL_LIBSYCLUTILS_INWORKLIST_

namespace sycl = cl::sycl;

class InWorklist {
    private:
    const gpu_size_t WORKLIST_CAPACITY;

    // GLOBAL ACCESSORS
    sycl::accessor<index_type, 1,
        sycl::access::mode::read,
        sycl::access::target::global_buffer>
            worklist;
    sycl::accessor<gpu_size_t, 1,
        sycl::access::mode::read,
        sycl::access::target::global_buffer>
            worklist_size;

    public:
    InWorklist(Pipe &pipe, sycl::handler &cgh)
        : WORKLIST_CAPACITY{ pipe.get_worklist_capacity() }
        , worklist{ pipe.get_in_worklist_buf(), cgh }
        , worklist_size{ pipe.get_in_worklist_size_buf(), cgh }
    { }

    /*
     * try to store the *index*th item of the worklist into node.
     *
     * Returns true iff was successful.
     *
     * NOTE: This isn't *really* const because it modifies the worklist,
     *       but we have to declare it as const for SYCL compilation
     */
    bool pop(gpu_size_t index, index_type &node) const {
        if(index >= worklist_size[0]) return false;
        node = worklist[index];
        return true;
    }

    /**
     * Utility function for printing (small) out-worklists
     */
    void print(const sycl::stream &stream) const {
        stream << "Capacity: " << WORKLIST_CAPACITY << "\n"
               << "size: " << worklist_size[0] << "\n";
        for(gpu_size_t i = 0; i < worklist_size[0]; ++i) {
            stream << worklist[i] << " ";
        }
        stream << sycl::endl;
    }
};

#endif
