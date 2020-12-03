/*  -*- mode: c++ -*- */
#include <CL/sycl.hpp>

// index_type
#include "sycl_csr_graph.h"
// Pipe gpu_size_t
#include "pipe.h"

#ifndef BREADTHNPAGEINSYCL_LIBSYCLUTILS_OUTWORKLIST_
#define BREADTHNPAGEINSYCL_LIBSYCLUTILS_OUTWORKLIST_

namespace sycl = cl::sycl;

// since we need atomics, use 32-bits for size
typedef uint32_t gpu_size_t ;

/**
 * An out-worklist has two parts:
 * a contiguous portion of items on the out-worklist,
 * and a partitioned portion where each work-group has its own space on the
 * worklist.
 *
 * [ contiguous portion ; group 0 portion ; group 1 portion ; .... ; group N portion ]
 *
 * A group's worklist offset is the start of its portion of the worklist.
 *
 * A group's worklist size is the number of entries in ITS PORTION of the worklist.
 *
 * Pushed entries are added to the group's portion of the worklist
 *
 * If you run out of space, use a Pipe to compress the group portions into
 * the contiguous portions.
 */
class OutWorklist {
    private:
        const gpu_size_t WORKLIST_CAPACITY,
                         NUM_WORK_GROUPS;

        // GLOBAL ACCESSORS
        sycl::accessor<index_type, 1,
            sycl::access::mode::read_write,
            sycl::access::target::global_buffer>
                worklist;
        sycl::accessor<gpu_size_t, 1,
            sycl::access::mode::atomic,
            sycl::access::target::global_buffer>
                worklist_sizes;
        sycl::accessor<gpu_size_t, 1,
            sycl::access::mode::read,
            sycl::access::target::global_buffer>
                worklist_offsets;
        // LOCAL ACCESSORS
        sycl::accessor<gpu_size_t, 1,
            sycl::access::mode::read_write,
            sycl::access::target::local>
                // start of my group's portion of the worklist
                my_offset,
                // first position after my group's portion of the worklist
                next_offset;
        sycl::accessor<gpu_size_t, 1,
            sycl::access::mode::atomic,
            sycl::access::target::local>
                my_size;
    public:
    OutWorklist(Pipe &pipe, sycl::handler &cgh)
    : WORKLIST_CAPACITY{ pipe.get_worklist_capacity() }
    , NUM_WORK_GROUPS{ pipe.get_num_work_groups() }
    , worklist{ pipe.get_out_worklist_buf(), cgh }
    , worklist_sizes{ pipe.get_out_worklist_sizes_buf(), cgh }
    , worklist_offsets{ pipe.get_out_worklist_offsets_buf(), cgh }
    , my_offset{ sycl::range<1>{1}, cgh }
    , next_offset{ sycl::range<1>{1}, cgh }
    , my_size  { sycl::range<1>{1}, cgh }
    { }

    /**
     * Set up local memory so that we can begin computing.
     *
     * If local memory is synchronized after this call,
     * only needs to be called by one thread in a group.
     *
     * NOTE: This isn't *really* const because it modifies local memory,
     *       but we have to declare it as const for SYCL compilation
     */
    void initializeLocalMemory(sycl::nd_item<1> my_item) const {
        my_offset[0] = worklist_offsets[my_item.get_group(0)];
        if(my_item.get_group(0) < NUM_WORK_GROUPS - 1) {
            next_offset[0] = worklist_offsets[my_item.get_group(0) + 1];
        }
        else {
            next_offset[0] = WORKLIST_CAPACITY;
        }
        my_size[0].store(worklist_sizes[my_item.get_group(0)].load());
    }

    /**
     * publish changes in local memory back to the global buffer
     *
     * local memory should be synchronized before this call.
     * only needs to be called by one thread in a group.
     *
     * NOTE: This isn't *really* const because it modifies the size,
     *       but we have to declare it as const for SYCL compilation
     */
    void publishLocalMemory(sycl::nd_item<1> my_item) const {
        worklist_sizes[my_item.get_group(0)].store(my_size[0].load());
    }

    /*
     * try to push node onto my_item's group portion of the out-worklist.
     *
     * Returns true iff was successful.
     *
     * ASSUMES initializeLocalMemory has been called by some
     * node in this group.
     *
     * finalizeLocalMemory must be called at some point after
     * this call.
     *
     * NOTE: This isn't *really* const because it modifies the worklist,
     *       but we have to declare it as const for SYCL compilation
     */
    bool push(const sycl::nd_item<1> &my_item, index_type node) const {
        gpu_size_t wl_index = my_offset[0] + my_size[0].fetch_add(1);
        // If full, return false and don't store
        if(wl_index >= next_offset[0]) {
            my_size[0].store(next_offset[0] - my_offset[0]);
            return false;
        }
        // Otherwise, we have room. store the node
        worklist[wl_index] = node;
        return true;
    }

    /**
     * Utility function for printing (small) out-worklists
     */
    void print(const sycl::stream &stream) const {
        stream << "Capacity: " << WORKLIST_CAPACITY << "\n"
               << "Contiguous region: size " << worklist_offsets[0] << "\n";
        for(gpu_size_t i = 0; i < worklist_offsets[0]; ++i) {
            stream << worklist[i] << " ";
        }
        for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
            stream << "\nWorklist " << wg << ":"
                   << "offset: " << worklist_offsets[wg]
                   << ", size: " << worklist_sizes[wg].load() << "\n";
            for(gpu_size_t j = 0; j < worklist_sizes[wg].load(); ++j) {
                stream << worklist[worklist_offsets[wg] + j] << " ";
            }
        }
        stream << sycl::endl;
    }
};

#endif
