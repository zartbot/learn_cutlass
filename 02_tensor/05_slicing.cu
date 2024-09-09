#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

#define MAXN 128 * 128

int main()
{
    // initial memory
    int *a_ptr = (int *)malloc(MAXN * sizeof(int));
    for (int i = 0; i < MAXN; ++i)
        a_ptr[i] = i;

    //(_3,_4,_5):(_20,_5,_1)
    Tensor A = make_tensor(a_ptr, make_shape(Int<3>{}, Int<4>{}, Int<5>{}),
                           GenRowMajor{});

    print_tensor(A);
    Tensor A1 = A(_, _, 2);
    print_tensor(A1);

    //(_3,_4),(_2,_4,_2)):((_64,_16),(_8,_2,_1)
    Tensor B = make_tensor(a_ptr, make_shape(make_shape(Int<3>{}, Int<4>{}), make_shape(Int<2>{}, Int<4>{}, Int<2>{})),
                           GenRowMajor{});

    print_tensor(B);
    Tensor C = B(make_coord(_, _), make_coord(1, 2, 1));
    print_tensor(C);

    Tensor D = B(make_coord(1, _), make_coord(0, _, 1));
    print_tensor(D);

    Tensor E = take<0,1>(B);
    print_tensor(E);

    Tensor F = take<0,1>(A);
    print_tensor(F);

}