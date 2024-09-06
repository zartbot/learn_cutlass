#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
    Layout a =  Layout<Shape<_4,Shape<Shape<_4,_5>,Shape<_6,_7>>>>{}; 
    printf("a :");
    print(a);

    printf("\nlayout<0>(a) :");
    print(layout<0>(a));

    printf("\nlayout<1>(a) :");
    print(layout<1>(a));

    printf("\nlayout<1,0>(a) :");
    print(layout<1,0>(a));

    printf("\nlayout<1,1>(a) :");
    print(layout<1,1>(a));

    printf("\nlayout<1,1,0>(a) :");
    print(layout<1,1,0>(a));

    Layout b= Layout<Shape<_2,_3,_5,_7>>{};  
    printf("\nb :");
    print(b);

    printf("\nselect<2>(b) :");
    print(select<2>(b));

    printf("\nselect<1,3>(b) :");
    print(select<1,3>(b));

    printf("\nselect<0,1,3>(b) :");
    print(select<0,1,3>(b));

    printf("\ntake<1,3>(b) :");
    print(take<1,3>(b));

    printf("\ntake<1,4>(b) :");
    print(take<1,4>(b));


}


Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)