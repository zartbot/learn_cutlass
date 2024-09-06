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

    auto layout_a = make_layout(make_shape(Int<4>{}, Int<2>{}),
                                make_stride(Int<1>{}, Int<16>{}));
    Tensor ta = make_tensor(A, layout_a);
    printf("\nLayout Tensor A: ");
    print_tensor(ta);

    auto layout_b= make_layout(make_shape(Int<6>{}),
                                make_stride(Int<1>{}));
    // auto layout_b = make_layout(make_shape(Int<2>{}, Int<4>{}),
    //                            make_stride(Int<4>{}, Int<2>{}));
    Tensor tb = make_tensor(A, layout_b);
    printf("\nLayout Tensor B: ");
    print_tensor(tb);

    Layout a_star = complement(layout_a, size(layout_a) * cosize(layout_b));
    Tensor ta_star = make_tensor(A, a_star);
    printf("\nLayout Tensor A* : ");
    print_tensor(ta_star);

    Layout a_star2 = composition(complement(layout_a, size(layout_a) * cosize(layout_b)), layout_b);
    Tensor ta_star2 = make_tensor(A, a_star2);
    printf("\nLayout Tensor A* o B: ");
    print_tensor(ta_star2);

    auto lp = logical_product(layout_a, layout_b);

    Tensor tlp = make_tensor(A, lp);
    printf("\nLayout Tensor Logical Product: ");
    print_tensor(tlp);
}

/*

// B-Tile < 3:3, (2,4):(1:8) >
    auto tiler = make_tile(Layout<_3, _3>{},     // Apply     3:3     to mode-0
                           Layout<Shape<_2, _4>, // Apply (2,4):(1,8) to mode-1
                                  Stride<_1, _8>>{});

                                     printf("\nLayout Tensor Logical Divide(mode-0): ");
    print_tensor(tensor<0>(tld));


    // ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
    auto zd = zipped_divide(layout_a, tiler);

    Tensor tzd = make_tensor(A, zd);
    printf("\nLayout Tensor Zipped Divide: ");
    print_tensor(tzd);
    printf("\nLayout Tensor Zipped Divide(mode-0): ");
    print_tensor(tensor<0>(tzd));
*/