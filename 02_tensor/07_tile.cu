#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

#define MAXN 128 * 128

#define PRINT(name, tensor) \
    printf("%8s : ", name); \
    print_tensor(tensor);   \
    print("\n");

int main()
{
    // initial memory
    int *hA = (int *)malloc(MAXN * sizeof(int));
    for (int i = 0; i < MAXN; ++i)
        hA[i] = i;

    // (4,6):(_1,4)
    Tensor A = make_tensor(hA, make_shape(4, 6));
    PRINT("A", A)
    auto tiler = Shape<_2, _3>{};

    //((_2,_3),(2,2)):((_1,4),(_2,12))
    Tensor tiled_a = zipped_divide(A, tiler);

    // inner
    int blockIdx_x = 0;
    int blockIdx_y = 1;
    Tensor cta_a = tiled_a(make_coord(_, _), make_coord(blockIdx_x, blockIdx_y));
    PRINT("CTA_A", cta_a)
    Tensor local_tileA = local_tile(A, tiler, make_coord(0, 1));
    PRINT("LOCAL_TILE", local_tileA)

    // outer
    int threadIdx_x = 3;
    Tensor thr_a = tiled_a(threadIdx_x, make_coord(_, _));
    PRINT("THR_A", thr_a)
    Tensor outer_partA = outer_partition(A, tiler, make_coord(1, 1));
    PRINT("OUTER_PART", outer_partA)
    Tensor local_partA = local_partition(A, make_layout(Shape<_2, _3>{}), 3);
    PRINT("LOCAL_PART", local_partA)

    free(hA);
}