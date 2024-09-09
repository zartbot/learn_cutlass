#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

#define MAXN 128 * 128

#define PRINTTENSOR(name, tensor) \
    printf("%20s : ", name);      \
    print(tensor);                \
    print("\n");

__global__ void tensor_kernel(float *A)
{
    // Untagged pointers
    Tensor tensor_8 = make_tensor(A, make_layout(Int<8>{})); // Construct with Layout
    Tensor tensor_8s = make_tensor(A, Int<8>{});             // Construct with Shape
    Tensor tensor_8d2 = make_tensor(A, 8, 2);                // Construct with Shape and Stride
    PRINTTENSOR("tensor_8", tensor_8)
    PRINTTENSOR("tensor_8s", tensor_8s)
    PRINTTENSOR("tensor_8d2", tensor_8d2)

    // Global memory (static or dynamic layouts)
    Tensor gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});
    Tensor gmem_8d = make_tensor(make_gmem_ptr(A), 8);
    Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));
    Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape(8, Int<16>{}),
                                     make_stride(Int<16>{}, Int<1>{}));
    PRINTTENSOR("gmem_8s", gmem_8s)
    PRINTTENSOR("gmem_8d", gmem_8d)
    PRINTTENSOR("gmem_8sx16d", gmem_8sx16d)
    PRINTTENSOR("gmem_8dx16s", gmem_8dx16s)

    // Shared memory (static or dynamic layouts)
    Layout smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
    __shared__ float smem[decltype(cosize(smem_layout))::value]; // (static-only allocation)

    Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
    Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), GenRowMajor{});
    PRINTTENSOR("smem_4x8_col", smem_4x8_col)
    PRINTTENSOR("smem_4x8_row", smem_4x8_row)
}

int main()
{
    // initial memory 
    float *A = (float *)malloc(MAXN * sizeof(float));
    for (int i = 0; i < MAXN; i++)
    {
        A[i] = float(i);
    }

    float *dA;
    cudaMalloc(&dA, MAXN * sizeof(float));
    cudaMemcpy(dA, A, MAXN * sizeof(float), cudaMemcpyHostToDevice);

    // Untagged pointers
    Tensor tensor_8 = make_tensor(A, make_layout(Int<8>{})); // Construct with Layout
    Tensor tensor_8s = make_tensor(A, Int<8>{});             // Construct with Shape
    Tensor tensor_8d2 = make_tensor(A, 8, 2);                // Construct with Shape and Stride
    PRINTTENSOR("host_tensor_8", tensor_8)
    PRINTTENSOR("host_tensor_8s", tensor_8s)
    PRINTTENSOR("host_tensor_8d2", tensor_8d2)
    printf("\n");

    tensor_kernel<<<1, 1>>>(dA);
    cudaDeviceSynchronize();
    free(A);
    cudaFree(dA);
}