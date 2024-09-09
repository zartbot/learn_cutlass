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

    //(_3,_4),(_2,_4,_2)):((_64,_16),(_8,_2,_1)
    Tensor B = make_tensor(a_ptr, make_shape(make_shape(Int<3>{}, Int<4>{}), make_shape(Int<2>{}, Int<4>{}, Int<2>{})),
                           GenRowMajor{});

    //(_3,_4,_2,_4,_2):(_64,_16,_8,_2,_1)
    Tensor C = flatten(B);
    print_tensor(C);

    // ((_3,_4),(_2,_4,_2)):((_1,_3),(_12,_24,_96))
    Tensor D = make_tensor(a_ptr, make_shape(make_shape(Int<3>{}, Int<4>{}), make_shape(Int<2>{}, Int<4>{}, Int<2>{})),
                           GenColMajor{});
    print_tensor(D);
    //_192:_1
    Tensor E = coalesce(D);
    print_tensor(E);

    //(_3,(_4,_2,_4),_2):(_64,(_16,_8,_2),_1):
    Tensor F = group_modes<1,4>(C);
    print_tensor(F);
}