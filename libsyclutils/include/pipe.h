/*  -*- mode: c++ -*- */
#include <CL/sycl.hpp>

// index_type
#include "sycl_csr_graph.h"

#ifndef BREADTHNPAGEINSYCL_LIBSYCLUTILS_PIPE_
#define BREADTHNPAGEINSYCL_LIBSYCLUTILS_PIPE_

#define THREAD_BLOCK_SIZE 256

namespace sycl = cl::sycl;

// since we need atomics, use 32-bits for size
typedef uint32_t gpu_size_t ;

class PIPETEST;
// classes used to name kernels
class InitializeWorklists;
class SwapWorklists;
class CompressOutWorklist;
class ResetOutWorklistOffsets;
class ClaimOwnership;
class DeDupe;

/**
 * Manages an in-worklist and an out-worklist of nodes.
 *
 * These worklists must hold integers in the range [0, NNODES)
 *
 * The out-worklist is implemented and maintained as described
 * in the SYCLOutWorklist class
 *
 */
class Pipe {
    private:
        const gpu_size_t WORKLIST_CAPACITY,
                         NUM_WORK_GROUPS,
                         NNODES;

        // GLOBAL MEMORY BUFFERS
        // in/out worklists
        sycl::buffer<index_type, 1> worklist1_buf,
                                    worklist2_buf,
                                    *in_worklist_buf = &worklist1_buf,
                                    *out_worklist_buf = &worklist2_buf;
        // size/offset of worklists (on a per-group basis for out-worklist)
        sycl::buffer<gpu_size_t, 1> in_worklist_size_buf,
                                    out_worklist_sizes_buf,
                                    out_worklist_offsets_buf;
        // used for de-duping out-worklist during compression.
        // Whenever a worker pushes a node onto the out-worklist, it should
        // for ownership.
        sycl::buffer<size_t, 1> owner_buf;

        /**
         * Used by compress to dedupe.
         */
        void dedupe(sycl::queue &queue);
    public:
        Pipe(gpu_size_t worklist_capacity,
             gpu_size_t nnodes,
             gpu_size_t num_work_groups)
            // round up to nearest multiple of num_work_groups plus num_work_groups
            // (the extra num_work_groups is so that if there are < worklist_capacity
            //  items are on the queue, each group gets at least 1 slot on the
            //  out-worklist)
            : WORKLIST_CAPACITY{ worklist_capacity + num_work_groups
                                 + (num_work_groups - (worklist_capacity) % num_work_groups)
                                   % num_work_groups}
            , NNODES{ nnodes }
            , NUM_WORK_GROUPS{ num_work_groups }
            , worklist1_buf{ sycl::range<1>{WORKLIST_CAPACITY} }
            , worklist2_buf{ sycl::range<1>{WORKLIST_CAPACITY} }
            , in_worklist_size_buf{ sycl::range<1>{1} }
            , out_worklist_sizes_buf{ sycl::range<1>{NUM_WORK_GROUPS} }
            , out_worklist_offsets_buf{ sycl::range<1>{NUM_WORK_GROUPS} }
            , owner_buf{ sycl::range<1>{NNODES} }
            { }

        /**
         * useful constant getters
         */ 
        gpu_size_t get_worklist_capacity() const {
            return this->WORKLIST_CAPACITY;
        }
        gpu_size_t get_num_work_groups() const {
            return this->NUM_WORK_GROUPS;
        }

        /**
         * out-worklist (and friends) getters
         */
        auto& get_out_worklist_buf() {
            return *(this->out_worklist_buf);
        }
        auto& get_out_worklist_sizes_buf() {
            return this->out_worklist_sizes_buf;
        }
        auto& get_out_worklist_offsets_buf() {
            return this->out_worklist_offsets_buf;
        }

        /**
         * in-worklist (and friends) getters
         */
        auto& get_in_worklist_buf() {
            return *(this->in_worklist_buf);
        }
        auto& get_in_worklist_size_buf() {
            return this->in_worklist_size_buf;
        }

        /**
         * Initialize the work-lists to empty.
         */ 
        void initialize(sycl::queue &queue) {
            queue.submit([&] (sycl::handler &cgh) {
                // copy constants
                const gpu_size_t NUM_WORK_GROUPS = this->NUM_WORK_GROUPS;
                const gpu_size_t WORKLIST_CAPACITY = this->WORKLIST_CAPACITY;
                // accessors
                auto in_worklist_size = this->in_worklist_size_buf.get_access<sycl::access::mode::write>(cgh);
                auto out_worklist_sizes = this->out_worklist_sizes_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto out_worklist_offsets = this->out_worklist_offsets_buf.get_access<sycl::access::mode::read_write>(cgh);

                cgh.single_task<class InitializeWorklists>([=]() {
                    in_worklist_size[0] = 0;
                    // reset the out-worklist sizes to 0 and offsets to
                    // evenly distributed throughout the worklist
                    for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                        out_worklist_sizes[wg] = 0;
                        out_worklist_offsets[wg] = wg * (WORKLIST_CAPACITY / NUM_WORK_GROUPS);
                    }
                });
            });
        }

        /**
         * Swap the in and out-worklists.
         *
         * ASSUMEs that the out-worklist has been compressed
         */
        void swapSlots(sycl::queue &queue) {
            std::swap(in_worklist_buf, out_worklist_buf);
            queue.submit([&] (sycl::handler &cgh) {
                // copy constants
                const gpu_size_t NUM_WORK_GROUPS = this->NUM_WORK_GROUPS;
                const gpu_size_t WORKLIST_CAPACITY = this->WORKLIST_CAPACITY;
                // accessors
                auto in_worklist_size = this->in_worklist_size_buf.get_access<sycl::access::mode::write>(cgh);
                auto out_worklist_sizes = this->out_worklist_sizes_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto out_worklist_offsets = this->out_worklist_offsets_buf.get_access<sycl::access::mode::read_write>(cgh);

                cgh.single_task<class SwapWorklists>([=]() {
                    // since out-worklist has been compressed, total size is the first offset
                    in_worklist_size[0] = out_worklist_offsets[0];
                    // reset the out-worklist sizes to 0 and offsets to
                    // evenly distributed throughout the    
                    for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                        out_worklist_sizes[wg] = 0;
                        out_worklist_offsets[wg] = wg * (WORKLIST_CAPACITY / NUM_WORK_GROUPS);
                    }
                });
            });
        }

    /**
     * Compress the out-worklist into a contiguous array
     *
     * De-dupes any entry (in the non-contiguous portion)
     * which appears more than once
     */ 
    void compress(sycl::queue &queue) {
        /// First, de-dupe
        this->dedupe(queue);
        /// Next, submit a job to copy memory from each group's portion of 
        // the out-worklist into the contiguous portion of the out-worklist
        queue.submit([&] (sycl::handler &cgh) {
            // copy constants
            const gpu_size_t NUM_WORK_GROUPS = this->NUM_WORK_GROUPS;
            const gpu_size_t WORKLIST_CAPACITY = this->WORKLIST_CAPACITY;
            // global accessors
            auto out_worklist = this->out_worklist_buf->get_access<sycl::access::mode::read_write>(cgh);
            auto out_worklist_sizes = this->out_worklist_sizes_buf.get_access<sycl::access::mode::read>(cgh);
            auto out_worklist_offsets = this->out_worklist_offsets_buf.get_access<sycl::access::mode::read>(cgh);

            // local accessors
            sycl::accessor<gpu_size_t, 1,
                           sycl::access::mode::read_write,
                           sycl::access::target::local>
                               // where to start writing my part of the worklist
                               compressed_start{sycl::range<1>{1}, cgh},
                               // offset of this group's part of the worklist
                               offset{sycl::range<1>{1}, cgh},
                               // size of this group's part of the worklist
                               size{sycl::range<1>{1}, cgh};

            const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                             NUM_WORK_ITEMS  = WORK_GROUP_SIZE * NUM_WORK_GROUPS;
            cgh.parallel_for<class CompressOutWorklist>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                          sycl::range<1>{WORK_GROUP_SIZE}},
            [=](sycl::nd_item<1> my_item) {
                // figure out where the stuff in my portion of the worklist needs to go,
                // and figure out my new offset and start
                gpu_size_t start;
                if(my_item.get_local_id()[0] == 0) {
                    start = 0;
                    for(size_t wg = 0; wg < my_item.get_group(0); ++wg) {
                        start += out_worklist_sizes[wg];
                    }
                    compressed_start[0] = start;
                    // load in offset and size from global memory
                    offset[0] = out_worklist_offsets[my_item.get_group(0)];
                    size[0] = out_worklist_sizes[my_item.get_group(0)];
                }
                // wait for thread 0 in my block to figure out our starting place
                my_item.barrier(sycl::access::fence_space::local_space);
                // Now work as a group to move our portion of the worklist into the contiguous
                // portion of the worklist
                start = compressed_start[0];
                gpu_size_t my_group_size = size[0],
                           my_group_offset = offset[0];
                for(gpu_size_t index = 0; index < my_group_size; index += WORK_GROUP_SIZE) 
                {
                    out_worklist[start + index] = out_worklist[my_group_offset + index];
                }
            });
        });
        /// Next, submit a job to reset the out-worklist sizes and offsets
        queue.submit([&] (sycl::handler &cgh) {
            // copy constants
            const gpu_size_t NUM_WORK_GROUPS = this->NUM_WORK_GROUPS;
            const gpu_size_t WORKLIST_CAPACITY = this->WORKLIST_CAPACITY;
            // accessors
            auto out_worklist_sizes = this->out_worklist_sizes_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto out_worklist_offsets = this->out_worklist_offsets_buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.single_task<class ResetOutWorklistOffsets>([=]() {
                // figure out how much we need to increase the first offset by,
                // and reset the sizes
                gpu_size_t first_offset_increase = 0;
                for(gpu_size_t wg = 0; wg < NUM_WORK_GROUPS; ++wg) {
                    first_offset_increase += out_worklist_sizes[wg];
                    out_worklist_sizes[wg] = 0;
                }
                out_worklist_offsets[0] += first_offset_increase;
                // Now divide up the remaining space into parts of the worklist for each
                // work-group
                gpu_size_t between_offsets = (WORKLIST_CAPACITY - out_worklist_offsets[0]) / NUM_WORK_GROUPS,
                           offset = out_worklist_offsets[0];
                for(gpu_size_t wg = 1; wg < NUM_WORK_GROUPS; ++wg) {
                    offset += between_offsets;
                    out_worklist_offsets[wg] = sycl::min(offset, WORKLIST_CAPACITY);
                }
            });
        });
    }
};

void Pipe::dedupe(sycl::queue &queue) {
    /// First, have each node compete for ownership. Whoever owns
    // a node on the worklist has the one that is "not a duplicate"
    queue.submit([&](sycl::handler &cgh) {
        // global accessors
        auto owner = this->owner_buf.get_access<sycl::access::mode::write>(cgh);
        auto out_worklist = this->out_worklist_buf->get_access<sycl::access::mode::read>(cgh);
        auto out_worklist_sizes = this->out_worklist_sizes_buf.get_access<sycl::access::mode::read>(cgh);
        auto out_worklist_offsets = this->out_worklist_offsets_buf.get_access<sycl::access::mode::read>(cgh);
        // local accessors
        sycl::accessor<gpu_size_t, 1,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
                           // offset of this group's part of the worklist
                           offset{sycl::range<1>{1}, cgh},
                           // size of this group's part of the worklist
                           size{sycl::range<1>{1}, cgh};

        const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                         NUM_WORK_ITEMS  = WORK_GROUP_SIZE * NUM_WORK_GROUPS;
        cgh.parallel_for<class ClaimOwnership>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                                 sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // have worker 0 load my offset and size into local memory
            if(my_item.get_local_id()[0] == 0) {
                offset[0] = out_worklist_offsets[my_item.get_group(0)];
                size[0] = out_worklist_sizes[my_item.get_group(0)];
            }
            my_item.barrier(sycl::access::fence_space::local_space);
            // load offset and size into private memory
            gpu_size_t my_offset = offset[0],
                       my_size = size[0];
            // compete for ownership
            for(gpu_size_t i = my_item.get_local_id()[0]; i < my_size; i += WORK_GROUP_SIZE) {
                owner[out_worklist[my_offset + i]] = my_item.get_global_id()[0];
            }
        });
    });
    queue.wait_and_throw();
    std::cout << "FIRST HERE\n" << std::endl;
    queue.submit([&](sycl::handler &cgh) {
        sycl::stream sycl_stream(1024, 512, cgh);
        const gpu_size_t NNODES = this->NNODES;
        auto owner = this->owner_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.single_task<class PIPETEST>([=]() {
            sycl_stream << "OWNERS: \n";
            for(size_t i = 0; i < NNODES; ++i) {
                sycl_stream << owner[i] << " ";
            }
            sycl_stream << sycl::endl;
        });
    });
    queue.wait_and_throw();
    std::cout << "HERE\n" << std::endl;
    /// Next, de-dupe
    queue.submit([&](sycl::handler &cgh) {
        // global accessors
        auto owner = this->owner_buf.get_access<sycl::access::mode::read>(cgh);
        auto out_worklist = this->out_worklist_buf->get_access<sycl::access::mode::read_write>(cgh);
        auto out_worklist_sizes = this->out_worklist_sizes_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto out_worklist_offsets = this->out_worklist_offsets_buf.get_access<sycl::access::mode::read>(cgh);
        // local accessors
        sycl::accessor<gpu_size_t, 1,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
                           // offset of this group's part of the worklist
                           offset{sycl::range<1>{1}, cgh},
                           // size of this group's part of the worklist
                           size{sycl::range<1>{1}, cgh};
        sycl::accessor<gpu_size_t, 1,
                       sycl::access::mode::atomic,
                       sycl::access::target::local>
                           // number of dupes de-duped so far
                           num_dedupes{sycl::range<1>{1}, cgh},
                           // where to substitute from when we found a dupe
                           wl_substitute_index{sycl::range<1>{1}, cgh};

        const gpu_size_t WORK_GROUP_SIZE = THREAD_BLOCK_SIZE,
                         NUM_WORK_ITEMS  = WORK_GROUP_SIZE * NUM_WORK_GROUPS;
        cgh.parallel_for<class DeDupe>(sycl::nd_range<1>{sycl::range<1>{NUM_WORK_ITEMS},
                                                         sycl::range<1>{WORK_GROUP_SIZE}},
        [=](sycl::nd_item<1> my_item) {
            // have worker 0 load my offset and size into local memory, and set num de-dupes to zero
            if(my_item.get_local_id()[0] == 0) {
                offset[0] = out_worklist_offsets[my_item.get_group(0)];
                size[0] = out_worklist_sizes[my_item.get_group(0)];
                num_dedupes[0].store(0);
                wl_substitute_index[0].store(offset[0] + size[0] - 1);
            }
            my_item.barrier(sycl::access::fence_space::local_space);

            // load offset and size into private memory
            gpu_size_t my_offset = offset[0],
                       my_size = size[0];
            // look to see if I am responsible for any dupes 
            for(gpu_size_t i = my_item.get_local_id()[0];
                i < my_size - num_dedupes[0].load();
                i += WORK_GROUP_SIZE)
            {
                // If I am a dupe, increase the dupe count
                if(owner[out_worklist[my_offset + i]] != my_item.get_global_id()[0]) {
                    num_dedupes[0].fetch_add(1);
                }
                // Try to substitute a non-dupe from the end of the list
                size_t desired_owner = my_item.get_global_id()[0];
                size_t wl_index = my_offset + i;
                while(owner[out_worklist[wl_index]] != desired_owner) {
                    wl_index = wl_substitute_index[0].fetch_sub(1);
                    // If the substitution index is out of bounds or
                    // not an index after me, then there was no valid
                    // substitute.
                    if(wl_index <= my_offset + i || wl_index >= my_offset + my_size) {
                        break;
                    }
                    // Otherwise, figure out who would own the substitue if the
                    // substitue were not a dupe
                    desired_owner = (wl_index - my_offset - i
                                     + my_item.get_global_id()[0]) % WORK_GROUP_SIZE;
                }
            }
            // Once everyone is done in my group, reduce the group size appropriately!
            my_item.barrier(sycl::access::fence_space::local_space);
            if(my_item.get_local_id()[0] == 0) {
                out_worklist_sizes[my_item.get_group(0)] -= num_dedupes[0].load();
            }
        });
    });
}

#endif
