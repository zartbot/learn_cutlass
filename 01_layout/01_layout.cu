

#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

#define PRINT(name, content)      \
    print(name);                  \
    print(" : ");                 \
    print(content);               \
    print(" Shape: ");            \
    print(cute::shape(content));  \
    print(" Stride: ");           \
    print(cute::stride(content)); \
    print(" rank: ");             \
    print(cute::rank(content));   \
    print(" depth: ");            \
    print(cute::depth(content));  \
    print(" size: ");             \
    print(cute::size(content));   \
    print(" cosize: ");           \
    print(cute::cosize(content)); \
    print("\n");


int main()
{

    Layout a = make_layout(make_shape(_6{}, _2{}), make_stride(_1{}, _7{}));
    Layout b = make_layout(make_shape(_3{}, _2{}), make_stride(_2{}, _3{}));
    Layout c = composition(a, b);
    Layout d = complement(a, c);
    Layout e = make_layout(a, c);

    PRINT("a", a);
    PRINT("b", b);
    PRINT("c", c);
    PRINT("c-get<1>", get<1>(c));
    PRINT("d", d);
    PRINT("e", e);

    Layout f_col = make_layout(make_shape(Int<2>{},3,4,5,6),
                               LayoutLeft{});
    Layout f_row = make_layout(make_shape(Int<2>{},3,4,5,6),
                               LayoutRight{});
    PRINT("fcol", f_col);
    PRINT("frow", f_row);
                  
    
}
