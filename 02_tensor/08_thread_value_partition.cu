#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

#define MAXN 128 * 128

int main()
{
    // initial memory
    int *hA = (int *)malloc(MAXN * sizeof(int));
       for (int i = 0; i < MAXN; ++i)
        hA[i] = i;
  

    // Construct a TV-layout that maps 8 thread indices and 4 value indices
    //   to 1D coordinates within a 4x8 tensor
    // (T8,V4) -> (M4,N8)
    auto tv_layout = Layout<Shape<Shape<_2, _4>, Shape<_2, _2>>,
                            Stride<Stride<_8, _1>, Stride<_4, _16>>>{}; // (8,4)
    print_layout(tv_layout);

    Tensor A = make_tensor(hA,make_shape(_4{},_8{}),GenColMajor{});
    print_tensor(A);
    // Compose A with the tv_layout to transform its shape and order
    Tensor tv = composition(A, tv_layout); // (8,4)
    // Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
    int tid = 1;
    Tensor v = tv(tid, _); // (4)
    print_tensor(v);
}

