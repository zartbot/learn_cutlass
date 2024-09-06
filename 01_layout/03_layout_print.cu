#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const &layout)
{
    for (int m = 0; m < size<0>(layout); ++m)
    {
        for (int n = 0; n < size<1>(layout); ++n)
        {
            printf("%3d  ", layout(m, n));
        }
        printf("\n");
    }
}

int main()
{

    Layout s46_col = make_layout(make_shape(Int<4>{}, 6),
                                   LayoutLeft{});
    Layout s46_row = make_layout(make_shape(Int<4>{}, 6),
                                   LayoutRight{});
/*
    printf("2d-col-major layout\n");
    print2D(s46_col);
    printf("2d-row-major layout\n");
    print2D(s46_row);

    print_layout(s46_col);

    print_latex(s46_col);

    */
   auto coord = make_coord(2,3);
   int inner_product = (int)get<0>(coord)  * (int)stride<0>(s46_col) +  
                       (int)get<1>(coord)  * (int)stride<1>(s46_col) ;
   printf("%3d %3d\n",s46_col(coord), s46_col(inner_product));

}