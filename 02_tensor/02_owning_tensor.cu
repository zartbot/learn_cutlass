#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

#define PRINTTENSOR(name, tensor) \
    printf("%20s : ", name);      \
    print(tensor);                \
    print("\n");

__global__ void tensor_kernel()
{
      // Register memory (static layouts only)
    Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
    Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{},
                                             LayoutRight{});
    Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{},
                                             Stride<_32, _2>{});
    Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
    PRINTTENSOR("rmem_4x8_col", rmem_4x8_col)
    PRINTTENSOR("rmem_4x8_row", rmem_4x8_row)
    PRINTTENSOR("rmem_4x8_pad", rmem_4x8_pad)
    PRINTTENSOR("rmem_4x8_like", rmem_4x8_like)
}

int main()
{
    // Register memory (static layouts only)
    Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
    Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{},
                                             LayoutRight{});
    Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{},
                                             Stride<_32, _2>{});
    Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
    PRINTTENSOR("host_rmem_4x8_col", rmem_4x8_col)
    PRINTTENSOR("host_rmem_4x8_row", rmem_4x8_row)
    PRINTTENSOR("host_rmem_4x8_pad", rmem_4x8_pad)
    PRINTTENSOR("host_rmem_4x8_like", rmem_4x8_like)
    printf("\n");

    tensor_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}