#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

int main()
{

    Tensor A = make_tensor<float>(Shape<Shape<_4, _5>, Int<13>>{},
                                  Stride<Stride<_12, _1>, _64>{});
    float *b_ptr = (float *)malloc(13 * 20 * sizeof(float));
    Tensor B = make_tensor(b_ptr, make_shape(13, 20));

    // Fill A via natural coordinates op[]
    for (int m0 = 0; m0 < size<0, 0>(A); ++m0)
        for (int m1 = 0; m1 < size<0, 1>(A); ++m1)
            for (int n = 0; n < size<1>(A); ++n)
                A[make_coord(make_coord(m0, m1), n)] = n + 2 * m0;

    // Transpose A into B using variadic op()
    for (int m = 0; m < size<0>(A); ++m)
        for (int n = 0; n < size<1>(A); ++n)
            B(n, m) = A(m, n);

    // Copy B to A as if they are arrays
    for (int i = 0; i < A.size(); ++i)
        A[i] = B[i];
    
    print_tensor(A);
    print_tensor(B);

    free(b_ptr);
}