#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
    auto shape = Shape<_3, Shape<_5, _4>>{};

    printf("\nidx2crd 19 : "); 
    print(idx2crd(19, shape)); 

    printf("\nidx2crd (1,5) : "); 
    print(idx2crd(make_coord(1, 5), shape));  
    
    printf("\nidx2crd (1,(1,2)) : "); 
    print(idx2crd(make_coord(1, make_coord(1, 2)), shape));   

    printf("\ncrd2idx (1,5) : ");
    print(crd2idx(make_coord(1, 5), shape));printf("\n"); 
}

/*

    Layout s46_col = make_layout(make_shape(Int<4>{}, 6),
                                 LayoutLeft{});
    print_layout(s46_col);
    //    print_latex(s46_col);

    for (int m = 0; m < size<0>(s46_col); ++m)
    {
        for (int n = 0; n < size<1>(s46_col); ++n)
        {
            auto coord = make_coord(m, n);
            int inner_product = (int)get<0>(coord) * (int)stride<0>(s46_col) +
                                (int)get<1>(coord) * (int)stride<1>(s46_col);
            printf("Coord[%3d,%3d]: %3d | verify %d\n", m, n, s46_col(coord),  s46_col(coord)==s46_col(inner_product));
        }
        printf("\n");
    }

    */