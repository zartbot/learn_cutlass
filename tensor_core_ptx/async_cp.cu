#include <stdio.h>
#include <stdint.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

__global__ void testcopy2(float *global1, float *global2, int subset_count)
{
    extern __shared__ float shared[];
    auto group = cooperative_groups::this_thread_block();

    // Create a synchronization object 
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (group.thread_rank() == 0)
    {
        init(&barrier, group.size());
    }
    group.sync();

    for (size_t subset = 0; subset < subset_count; ++subset)
    {
        cuda::memcpy_async(group, shared,
                           &global1[subset * group.size()], sizeof(float) * group.size(), barrier);
        cuda::memcpy_async(group, shared + group.size(),
                           &global2[subset * group.size()], sizeof(float) * group.size(), barrier);

        barrier.arrive_and_wait(); // Wait for all copies to complete

        // simulate compute
        if (group.thread_rank() == 0)
        {
            printf("%f ", shared[0]);
        }

        barrier.arrive_and_wait();
    }
}

/*
__global__ void testcopy(float *x, int N) {
    int tid = threadIdx.x;
    __shared__ float Tile[32];
    *reinterpret_cast<float4*>(&Tile[tid]) = *reinterpret_cast<float4*>(&x[tid*4]);
    printf("%f ", Tile[tid]);
}*/

/*
__global__ void testcopy2(float *x, int N) {
    int tid = threadIdx.x;
    __shared__ float Tile[32];
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&Tile[tid])),
                "l"(&x[tid]),
                "n"(16)
            );

    printf("%f ", Tile[tid]);
}
*/

int main()
{
    const int N_DATA = 1024;
    float *x;
    cudaMalloc(&x, N_DATA * sizeof(float));
    float *y;
    cudaMalloc(&y, N_DATA * sizeof(float));

    dim3 gridDim(32, 1, 1);
    dim3 blockDim(32, 1, 1);
    testcopy2<<<gridDim, blockDim>>>(x, y, N_DATA);

    cudaFree(x);
    cudaDeviceReset();

    return 0;
}
