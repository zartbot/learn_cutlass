#include <getopt.h>
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

#define MAXN 128 * 128
template <class T>
auto make_composition(T a, int N, int r)
{
    auto shape_b = make_shape(N);
    auto stride_b = make_stride(r);
    Layout b = make_layout(shape_b, stride_b);
    printf("\nLayout-B: ");
    print(b);
    auto c = composition(a, b);
    printf("\nLayout-A o B: ");
    print(c);
    printf("\n");
    return c;
}

int main()
{


    auto sa = make_shape(Int<36>{}, Int<18>{});
    auto a = make_layout(sa, make_stride(Int<1>{}, Int<72>{}));

    auto sb = make_shape(Int<9>{}, Int<4>{});
    auto b = make_layout(sb, make_stride(Int<4>{}, Int<9>{}));
    
    auto c = composition(a, b);
    print(c);







    auto s2 = make_shape(Int<4>{}, Int<6>{}, Int<8>{},Int<10>{});
    auto a2 = make_layout(s2, make_stride(Int<2>{}, Int<3>{}, Int<5>{},Int<7>{}));
    auto b2 = make_layout(make_shape(Int<6>{}), make_stride(Int<12>{}));
    auto c2 = composition(a2, b2);
    print(c2);


    
}