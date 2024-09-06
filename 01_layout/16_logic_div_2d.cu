#include <getopt.h>
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

#define MAXN 128 * 128

int main()
{
    // initial memory with physical layout
    int *A = (int *)malloc(MAXN * sizeof(int));
    for (int i = 0; i < MAXN; i++)
    {
        A[i] = int(i);
    }

    // A: shape is (9,32)
    auto layout_a = make_layout(make_shape(Int<9>{}, make_shape(Int<4>{}, Int<8>{})),
                                make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
    Tensor ta = make_tensor(A, layout_a);
    printf("\nLayout Tensor A: ");
    print_tensor(ta);

    // B-Tile < 3:3, (2,4):(1:8) >
    auto tiler = make_tile(Layout<_3, _3>{},     // Apply     3:3     to mode-0
                           Layout<Shape<_2, _4>, // Apply (2,4):(1,8) to mode-1
                                  Stride<_1, _8>>{});

    // ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))
    auto ld = logical_divide(layout_a, tiler);

    Tensor tld = make_tensor(A, ld);
    printf("\nLayout Tensor Logical Divide: ");
    print_tensor(tld);
     printf("\nLayout Tensor Logical Divide(mode-0): ");
    print_tensor(tensor<0>(tld));


    // ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
    auto zd = zipped_divide(layout_a, tiler);

    Tensor tzd = make_tensor(A, zd);
    printf("\nLayout Tensor Zipped Divide: ");
    print_tensor(tzd);
    printf("\nLayout Tensor Zipped Divide(mode-0): ");
    print_tensor(tensor<0>(tzd));
}