#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
    auto shape = Shape<_4, Shape<_2, _4>>{};
    auto stride = Stride<_2,Stride<_1,_8>>{};
    auto l = make_layout(shape,stride);
    print_layout(l);

    printf("\ncrd2idx 22 : "); 
    print(crd2idx(22, shape, stride)); 

    printf("\ncrd2idx (2,5) : "); 
    print(crd2idx(make_coord(2,5), shape, stride)); 

    printf("\ncrd2idx (2,(1,2)) : "); 
    print(crd2idx(make_coord(2,make_coord(1,2)), shape, stride)); 

    printf("\n");

}

