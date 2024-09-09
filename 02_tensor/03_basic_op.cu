#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

#define MAXN 128 * 128

#define PRINT(name, tensor)  \
    printf("%20s : ", name); \
    print(tensor);           \
    print("\n");

__global__ void tensor_kernel(float *A)
{
    Tensor t = make_tensor(A, make_shape(_8{}, _4{}), GenColMajor{});
    PRINT("tensor_8x4", t)
    PRINT("Layout", t.layout())
    PRINT("SHAPE", t.shape())
    PRINT("STRIDE", t.stride())
    PRINT("SIZE", t.size())
    PRINT("Data", t.data())
    PRINT("Rank", t.rank)
    PRINT("Depth", depth(t))
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

    tensor_kernel<<<1, 1>>>(dA);
    cudaDeviceSynchronize();
    free(A);
    cudaFree(dA);
}