#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
    Layout a = Layout<_3,_1>{};                     // 3:1
    Layout b = Layout<_4,_3>{};                     // 4:3

    Layout row = make_layout(a, b);                 // (3,4):(1,3)
    printf("\nrow = make_layout(a, b) :");
    print(row);

    Layout col = make_layout(b, a);                 // (4,3):(3,1)
    printf("\ncol = make_layout(b, a) :");
    print(col);

    Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
    printf("\nq = make_layout(row, col); :");
    print(q);

    auto sc = Shape<_3,_4,_5>();
    Layout c = make_layout(sc,LayoutLeft{});
    printf("\nc :");
    print(c);
   
    auto sd = Shape<_6,_7,_8>();
    Layout d = make_layout(sd,LayoutLeft{});
    printf("\nd :");
    print(d);

    Layout cd = make_layout(sc,sd);
    printf("\nmake_layout(c,d); :");
    print(cd);
    

Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))


}



