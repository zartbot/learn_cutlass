#include <cuda.h>         // CUtensormap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__global__ void kernel(const CUtensorMap tensor_map, int x, int y) {

  const int tid = threadIdx.x;
  // bluk tensor 的拷贝操作需要 Shared Memory 首地址对齐 128 字节。
  __shared__ alignas(128) int smem_buffer[128][128];

  // 创建 Shared Memory 的 cuda::barrier 变量 
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // 初始化 barrier 
    init(&bar, blockDim.x);
    // 插入 fence
    cde::fence_proxy_async_shared_cta();    
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // 发起 TMA 二维异步拷贝操作
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // 设置同步等待点，指定需要等待的拷贝完成的字节数。
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // 等待完成拷贝
  bar.wait(std::move(token));

  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // 插入 fence
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }

  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}