#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

#define MAXN 128*128

using namespace cute;

#define PRINTTENSOR(name,  tensor) \
    printf("\nTensor : %s :",name);                 \
    print_tensor(tensor);                 \
    print("\n");      

int main()
{
    // initial memory with physical layout
    int* A = (int*)malloc(MAXN * sizeof(int));
    for(int i =0 ; i < MAXN ; i++){
	    A[i]=int(i);
    }   
    
    // 2D tensor
    auto shape2d = make_shape(_4{},_8{});

    //(_4,_8):(_1,_4)
    Layout l1 = make_layout(shape2d, LayoutLeft{});
    Tensor t1 = make_tensor(A, l1);
    PRINTTENSOR("LayoutLeft",t1)

    //(_4,_8):(_8,_1)
    Layout l2 = make_layout(shape2d, LayoutRight{});
    Tensor t2 = make_tensor(A, l2);
    PRINTTENSOR("LayoutRight",t2)
    
    //(_4,_8):(_3,_2)
    Layout l3 = make_layout(shape2d, make_stride(_3{},_2{}));
    Tensor t3 = make_tensor(A, l3);
    PRINTTENSOR("(_4,_8):(_3,_2)",t3)
}


/*

    auto layout = Layout<Shape<_16, _16>,
                       Stride<_16, _1>>{};
    auto swizzled_layout = composition(Swizzle<2,0,3>{}, layout);
    Tensor s_2d = make_tensor(A, swizzled_layout);
    PRINTTENSOR("2d swizzled_layout", swizzled_layout, s_2d)
    //print_latex(swizzled_layout);



    // 3D tensor
    auto shape3 = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto stride3 = make_stride(Int<12>{}, Int<4>{}, Int<1>{});
    auto layout3 = make_layout(shape3, stride3);
    Tensor t_3d = make_tensor(A,layout3);
    PRINTTENSOR("3d", layout3, t_3d);

*/