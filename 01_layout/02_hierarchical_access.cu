

#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

#define PRINT_LAYOUT(name, content) \
    print(name);                    \
    print(" : ");                   \
    print(content);                 \
    print(" Shape: ");              \
    print(cute::shape(content));    \
    print(" Stride: ");             \
    print(cute::stride(content));   \
    print(" rank: ");               \
    print(cute::rank(content));     \
    print(" depth: ");              \
    print(cute::depth(content));    \
    print(" size: ");               \
    print(cute::size(content));     \
    print(" cosize: ");             \
    print(cute::cosize(content));   \
    print("\n");

int main()
{

    auto s1 = make_shape(_1{}, _2{});
    auto d1 = make_stride(_1{}, _2{});
    auto s2 = make_shape(_2{}, _3{}, s1);
    auto d2 = make_stride(_2{}, _3{}, d1);
    auto s3 = make_shape(_3{}, _4{}, _5{}, s2);
    auto d3 = make_stride(_3{}, _4{}, _5{}, d2);
    auto s4 = make_shape(_4{}, _5{}, _6{}, s3);
    auto d4 = make_stride(_4{}, _5{}, _6{}, d3);
    auto s5 = make_shape(_5{}, _6{}, _7{},_8{}, s4);
    auto d5 = make_stride(_5{}, _6{}, _7{},_8{}, d4);

    Layout a = make_layout(s5, d5);
    PRINT_LAYOUT("a", a);
    PRINT_LAYOUT("a<4>", get<4>(a));
    auto a43 = get<4,3>(a);
    PRINT_LAYOUT("a<4,3>",a43 );
    auto a433 = get<4,3,3>(a);
    PRINT_LAYOUT("a<4,3,3>",a433 );
    auto a4332 = get<4,3,3,2>(a);
    PRINT_LAYOUT("a<4,3,3,2> ",a4332 );
    auto a4332_1 = get<2>(a433);
    PRINT_LAYOUT("a<4,3,3,2>1",a4332_1 );
}
