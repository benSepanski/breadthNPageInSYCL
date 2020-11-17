# Lonestar CUDA implementation of BFS

Lonestar documentation [here](https://iss.oden.utexas.edu/?p=projects/galois/analytics/gpu-bfs),
a little more documenation [here](https://iss.oden.utexas.edu/?p=projects/galois/analytics/gpu-bfs).
Optimizations are described in the [IrGL paper](https://dl.acm.org/doi/10.1145/3022671.2984015)

You should really read this file from the bottom up.

## Imports and constants

The ThreadWork struct comes from ["thread\_work.h"](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/include/thread_work.h)
in [libgpu](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/libgpu/).
The [Worklist2 struct](https://github.com/IntelligentSoftwareSystems/Galois/blob/357f258718682ee2058ccf0d57557a124345ece9/libgpu/include/worklist.h)
also comes from libgpu.

`start_node` comes from the ["support.cu"](https://github.com/IntelligentSoftwareSystems/Galois/blob/master/lonestar/analytics/gpu/bfs/support.cu)
for bfs.

```Cuda
/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "thread_work.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=True $ backoff_blocking_factor=4 $ parcomb=True $ np_schedulers=set(['fg', 'tb', 'wp']) $ cc_disable=set([]) $ tb_lb=True $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ cuda.worklist_type=basic";
struct ThreadWork t_work;
extern int start_node;
bool enable_lb = false;
typedef int edge_data_type;
typedef int node_data_type;
extern const node_data_type INF = INT_MAX;
static const int __tb_bfs_kernel = TB_SIZE;
static const int __tb_gg_main_pipe_1_gpu_gb = 256;
```

## BFS Initialization

Set `node_data` of source to `0` and all others to `INF`.

```Cuda
__global__ void bfs_init(CSRGraph graph, int src)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type node_end;
  node_end = (graph).nnodes;
  for (index_type node = 0 + tid; node < node_end; node += nthreads)
  {
    graph.node_data[node] = (node == src) ? 0 : INF ;
  }
}
```

## BFS Load-balancer

* `compute_src_and_offset` is defined in the ThreadWork
  class [here](https://github.com/IntelligentSoftwareSystems/Galois/blob/158b572802864bedc2db6b6d3c37d1fdd5035886/libgpu/include/thread_work.h#L54).

* Repeatedly does the following:
    
    - Has thread 0 of the block
      figure out the block start index and block end index
    - If we haven't done all out work yet:
        - Get the thread's source node and edge offset `offset`
        - Get the `offset`th edge of the source node
        - If that destination node is newly discovered,
          put it on the queue and mark it as on this level


```Cuda
__global__ void bfs_kernel_dev_TB_LB(CSRGraph graph, int LEVEL, int * thread_prefix_work_wl, unsigned int num_items, PipeContextT<Worklist2> thread_src_wl, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  __shared__ unsigned int total_work;
  __shared__ unsigned block_start_src_index;
  __shared__ unsigned block_end_src_index;
  unsigned my_work;
  unsigned node;
  unsigned int offset;
  unsigned int current_work;
  unsigned blockdim_x = BLOCK_DIM_X;
  total_work = thread_prefix_work_wl[num_items - 1];
  my_work = ceilf((float)(total_work) / (float) nthreads);

  __syncthreads();

  if (my_work != 0)
  {
    current_work = tid;
  }
  for (unsigned i =0; i < my_work; i++)
  {
    unsigned int block_start_work;
    unsigned int block_end_work;
    if (threadIdx.x == 0)
    {
      if (current_work < total_work)
      {
        block_start_work = current_work;
        block_end_work=current_work + blockdim_x - 1;
        if (block_end_work >= total_work)
        {
          block_end_work = total_work - 1;
        }
        block_start_src_index = compute_src_and_offset(0, num_items - 1,  block_start_work+1, thread_prefix_work_wl, num_items,offset);
        block_end_src_index = compute_src_and_offset(0, num_items - 1, block_end_work+1, thread_prefix_work_wl, num_items, offset);
      }
    }
    __syncthreads();

    if (current_work < total_work)
    {
      unsigned src_index;
      index_type edge;
      src_index = compute_src_and_offset(block_start_src_index, block_end_src_index, current_work+1, thread_prefix_work_wl,num_items, offset);
      node= thread_src_wl.in_wl().dwl[src_index];
      edge = (graph).getFirstEdge(node)+ offset;
      {
        index_type dst;
        dst = graph.getAbsDestination(edge);
        if (graph.node_data[dst] == INF)
        {
          index_type _start_24;
          graph.node_data[dst] = LEVEL;
          _start_24 = (out_wl).setup_push_warp_one();;
          (out_wl).do_push(_start_24, 0, dst);
        }
      }
      current_work = current_work + nthreads;
    }
  }
}
```

## Inspection Kernel

* `DEGREE_LIMIT` is defined [here](https://github.com/IntelligentSoftwareSystems/Galois/blob/158b572802864bedc2db6b6d3c37d1fdd5035886/libgpu/include/ggcuda.h)
  in `libgpu`
* `thread_<...>_wl.in_wl()` is a [worklist](https://github.com/IntelligentSoftwareSystems/Galois/blob/2a2e5656d739c6228c996ce2f952ec65216aa22f/libgpu/include/worklist.h)
* For each node (with index `wlnode` in the worklist) associated to this thread
    - `pop_id` stores the thread node-id in `node`
    - `pop` is true if `node` is a valid node and its degree is too bg
    - If `pop` is true, puts the node on `thread_src_wl` and its
      degree on `thread_work_wl`

```Cuda
__global__ void Inspect_bfs_kernel_dev(CSRGraph graph, int LEVEL, PipeContextT<Worklist2> thread_work_wl, PipeContextT<Worklist2> thread_src_wl, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  index_type wlnode_end;
  wlnode_end = *((volatile index_type *) (in_wl).dindex);
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    int index;
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) >= DEGREE_LIMIT)) ? true: false);
    if (pop)
    {
      index = thread_work_wl.in_wl().push_range(1) ;
      thread_src_wl.in_wl().push_range(1);
      thread_work_wl.in_wl().dwl[index] = (graph).getOutDegree(node);
      thread_src_wl.in_wl().dwl[index] = node;
    }
  }
}
```

## BFS Kernel without load balancing

Some of the notation refers to scheduling policies from the
[IrGL Paper](https://dl.acm.org/doi/10.1145/2983990.2984015)
(in section 4.3).
The 
`tb` refers to threadblock, `wp` to warp, and `fg` to fine-grained.
`np` refers to ``nested parallelism"
(I believe the notation comes from the [CUDA-NP paper](https://dl.acm.org/doi/10.1145/2555243.2555254), but it
for our purposes it is described in section 4 of the [IrGL Paper](https://dl.acm.org/doi/10.1145/2983990.2984015)).
and the relevant classes live in [libgpu/incldue/internal.h](https://github.com/IntelligentSoftwareSystems/Galois/blob/8fa18faea4cd0df855ae5d78a6aa2501c4cca6cd/libgpu/include/internal.h).
Look to section 4.3.1 for definitions
(and listing 10 and 11 to see how they are combined).
The `while(true)` loop in listing 11 corresponds to the
`while(true)` loop here, and inside
we see the scheduling combination described in listing 10.

Due to the length of this code, I am putting annotations in inline comments.
    
```CUDA
__device__ void bfs_kernel_dev(CSRGraph graph, int LEVEL, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D; // Global thread ID
  unsigned nthreads = TOTAL_THREADS_1D;  // total num threads

  const unsigned __kernel_tb_size = __tb_bfs_kernel;
  index_type wlnode_end;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;  // Note that this is shared
  wlnode_end = roundup((*((volatile index_type *) (in_wl).dindex)), (blockDim.x));
  // Each thread is assigned nodes in the waitlist according to thread-id
  for (index_type wlnode = 0 + tid; wlnode < wlnode_end; wlnode += nthreads)
  {
    int node;
    bool pop;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    // As long as you have a valid node (and if load-balancing is enabled,
    // if the out-degree is small enough that this node has not already 
    // been handled by the load-balancer)
    // get some nested parallelism info about your node, storing degree as
    // as _np.size and the first as as _np.start.
    pop = (in_wl).pop_id(wlnode, node) && ((( node < (graph).nnodes ) && ( (graph).getOutDegree(node) < DEGREE_LIMIT)) ? true: false);
    struct NPInspector1 _np = {0,0,0,0,0,0};
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    // TODO: what is this??
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    // FIRST: do thread-block scheduling ///////////////////////////////////////
    //
    // nps.tb.owner is the nested-parallelism thread-block owner.
    // Different threads will compete for thread-block ownership until
    // all threads have had a chance to get their inner loop handled
    // by thread-block.
    // It starts out as an invalid index.
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    __syncthreads();
    // threads repeatedly compete to have the thread-block work
    // on their inner loop
    while (true)
    {
      // Compete for ownership (if you have a big enough inner loop)
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      __syncthreads();
      // If no-one asked for ownership, we're done!
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        __syncthreads();
        break;
      }
      // If someone got ownership, have them load their information
      // into the shared nps object
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      __syncthreads();
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      // Relinquish ownership for next round
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      // Handle the thread-block owners inner-loop as a thread-block
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type edge;
        edge = ns +_np_j;
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          if (graph.node_data[dst] == INF)
          {
            index_type _start_24;
            graph.node_data[dst] = LEVEL;
            _start_24 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_24, 0, dst);
          }
        }
      }
      __syncthreads();
    }

    /// NEXT: Do warp-scheduling //////////////////////////////////////////////
    {
      const int warpid = threadIdx.x / 32;   // warp identifier
      const int _np_laneid = cub::LaneId();  // pos of thread in warp https://nvlabs.github.io/cub/group___util_ptx.html
      // The any_sync makes sure that if any thread needs to handle
      // some of its inner loop using a warp-schedule, then all threads
      // enter this loop. This ensures that no thread deadlocks on a barrier
      // that another thread fails to encounter.
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture__notes
      while (__any_sync(0xffffffff,_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB))
      {
        // If is of warp-schedule size, compete for ownership
        if (_np.size >= _NP_CROSSOVER_WP && _np.size < _NP_CROSSOVER_TB)
        {
          nps.warp.owner[warpid] = _np_laneid;
        }
        // One would expect a syncthreads here, but due to warp synchronization
        // we don't need one.
        if (nps.warp.owner[warpid] == _np_laneid)
        {
          nps.warp.start[warpid] = _np.start;
          nps.warp.size[warpid] = _np.size;

          _np.start = 0;
          _np.size = 0;
        }
        // Distribute inner loop across warp
        index_type _np_w_start = nps.warp.start[warpid];
        index_type _np_w_size = nps.warp.size[warpid];
        for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
        {
          index_type edge;
          edge = _np_w_start +_np_ii;
          {
            index_type dst;
            dst = graph.getAbsDestination(edge);
            if (graph.node_data[dst] == INF)
            {
              index_type _start_24;
              graph.node_data[dst] = LEVEL;
              _start_24 = (out_wl).setup_push_warp_one();;
              (out_wl).do_push(_start_24, 0, dst);
            }
          }
        }
      }
      __syncthreads();
    }

    /// FINALLY: fine-grained parallelism /////////////////////////////////////
    __syncthreads();
    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect(nps.fg.itvalue, ITSIZE);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst;
          dst = graph.getAbsDestination(edge);
          if (graph.node_data[dst] == INF)
          {
            index_type _start_24;
            graph.node_data[dst] = LEVEL;
            _start_24 = (out_wl).setup_push_warp_one();;
            (out_wl).do_push(_start_24, 0, dst);
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
  }
}
```

## Call BFS after worklist reset

```Cuda
__global__ void bfs_kernel(CSRGraph graph, int LEVEL, bool enable_lb, Worklist2 in_wl, Worklist2 out_wl)
{
  unsigned tid = TID_1D;

  if (tid == 0)
    in_wl.reset_next_slot();

  bfs_kernel_dev(graph, LEVEL, enable_lb, in_wl, out_wl);
}
```

## Load Balancing BFS Dispatcher

* While there are items in the pipeline:

    - Uses `t_work` which is of class [Thread Work](https://github.com/IntelligentSoftwareSystems/Galois/blob/2a2e5656d739c6228c996ce2f952ec65216aa22f/libgpu/include/thread_work.h) from libgpu

    - Runs the [inspection kernel](#inspection-kernel) with `t_work`

    - Uses `t_work` to run a load-balanced BFS round using
      the `bfs_kernel_dev_TB_LB` [kernel](#BFS-Load-balancer)
      on the high-degree nodes (as determined by the inspection kernel)

    - Calls to [bfs kernel](#bfs-kernel-without-load-balancing) through
      [this kernel](#Call-BFS-after-worklist-reset) on the non-high
      degree nodes (as determined by the inspection kernel)

```Cuda
void gg_main_pipe_1(CSRGraph& gg, int& LEVEL, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  while (pipe.in_wl().nitems())
  {
    pipe.out_wl().will_write();
    if (enable_lb)
    {
      t_work.reset_thread_work();
      Inspect_bfs_kernel_dev <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, t_work.thread_work_wl, t_work.thread_src_wl, enable_lb, pipe.in_wl(), pipe.out_wl());
      cudaDeviceSynchronize();
      int num_items = t_work.thread_work_wl.in_wl().nitems();
      if (num_items != 0)
      {
        t_work.compute_prefix_sum();
        cudaDeviceSynchronize();
        bfs_kernel_dev_TB_LB <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, t_work.thread_prefix_work_wl.gpu_wr_ptr(), num_items, t_work.thread_src_wl, pipe.in_wl(), pipe.out_wl());
        cudaDeviceSynchronize();
      }
    }
    bfs_kernel <<<blocks, __tb_bfs_kernel>>>(gg, LEVEL, enable_lb, pipe.in_wl(), pipe.out_wl());
    cudaDeviceSynchronize();
    pipe.in_wl().swap_slots();
    pipe.advance2();
    LEVEL++;
  }
}
```

## BFS dispatcher without load balancing

* launch bounds has a good description on this
[stack overflow post](https://stackoverflow.com/questions/44704506/limiting-register-usage-in-cuda-launch-bounds-vs-maxrregcount).

* Dispatches `bfs_kernel_dev` [kernel](#bfs-kernel-without-load-balancing)
  for each level.

```Cuda
__global__ void __launch_bounds__(__tb_gg_main_pipe_1_gpu_gb) gg_main_pipe_1_gpu_gb(CSRGraph gg, int LEVEL, PipeContextT<Worklist2> pipe, int* cl_LEVEL, bool enable_lb, GlobalBarrier gb)
{
  unsigned tid = TID_1D;

  LEVEL = *cl_LEVEL;
  while (pipe.in_wl().nitems())
  {
    if (tid == 0)
      pipe.in_wl().reset_next_slot();
    bfs_kernel_dev (gg, LEVEL, enable_lb, pipe.in_wl(), pipe.out_wl());
    pipe.in_wl().swap_slots();
    gb.Sync();
    pipe.advance2();
    LEVEL++;
  }
  gb.Sync();
  if (tid == 0)
  {
    *cl_LEVEL = LEVEL;
  }
}
```

## BFS dispatch wrapper

* If load-balancing, calls `gg_main_pipe_1`
  [kernel](#Load-Balancing-BFS-Dispatcher)

* Otherwise, invokes the `gg_main_pipe_1_gpu_gb`
  [kernel](#BFS-dispatcher-without-load-balancing)

```Cuda
void gg_main_pipe_1_wrapper(CSRGraph& gg, int& LEVEL, PipeContextT<Worklist2>& pipe, dim3& blocks, dim3& threads)
{
  static GlobalBarrierLifetime gg_main_pipe_1_gpu_gb_barrier;
  static bool gg_main_pipe_1_gpu_gb_barrier_inited;
  extern bool enable_lb;
  static const size_t gg_main_pipe_1_gpu_gb_residency = maximum_residency(gg_main_pipe_1_gpu_gb, __tb_gg_main_pipe_1_gpu_gb, 0);
  static const size_t gg_main_pipe_1_gpu_gb_blocks = GG_MIN(blocks.x, ggc_get_nSM() * gg_main_pipe_1_gpu_gb_residency);
  if(!gg_main_pipe_1_gpu_gb_barrier_inited) { gg_main_pipe_1_gpu_gb_barrier.Setup(gg_main_pipe_1_gpu_gb_blocks); gg_main_pipe_1_gpu_gb_barrier_inited = true;};
  if (enable_lb)
  {
    gg_main_pipe_1(gg,LEVEL,pipe,blocks,threads);
  }
  else
  {
    int* cl_LEVEL;
    check_cuda(cudaMalloc(&cl_LEVEL, sizeof(int) * 1));
    check_cuda(cudaMemcpy(cl_LEVEL, &LEVEL, sizeof(int) * 1, cudaMemcpyHostToDevice));

    gg_main_pipe_1_gpu_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(gg,LEVEL,pipe,cl_LEVEL, enable_lb, gg_main_pipe_1_gpu_gb_barrier);
    check_cuda(cudaMemcpy(&LEVEL, cl_LEVEL, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(cl_LEVEL));
  }
}
```

## gg\_main

* See [kernel sizing](https://github.com/IntelligentSoftwareSystems/Galois/blob/2a2e5656d739c6228c996ce2f952ec65216aa22f/libgpu/src/skelapp/skel.cu#L37)
for blocks/thread size.

* Calls [`bfs_init`](#bfs-initialization) then sychronizes.

* Puts start node on the worklist and calls the 
  [`gg_main_pipe_1_wrapper`](#bfs-dispatch-wrapper)

```Cuda
void gg_main(CSRGraph& hg, CSRGraph& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  t_work.init_thread_work(gg.nnodes);
  PipeContextT<Worklist2> wl;
  bfs_init <<<blocks, threads>>>(gg, start_node);
  cudaDeviceSynchronize();
  int LEVEL = 1;
  wl = PipeContextT<Worklist2>(gg.nnodes);
  wl.in_wl().wl[0] = start_node;
  wl.in_wl().update_gpu(1);
  gg_main_pipe_1_wrapper(gg,LEVEL,wl,blocks,threads);
}
```
