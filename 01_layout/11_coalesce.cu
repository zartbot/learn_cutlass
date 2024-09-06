#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;


template<class T>
void print_coalesce(T layout) {
    printf("H-Layout  :");
    print(layout);
    printf("\nCoalesce-Layout :");
    print(coalesce(layout));
    printf("\n");
}

int main()
{
    Layout a0 = make_layout(Shape<_2,_4>{},Stride<_1,_2>{});
    print_coalesce(a0);
    
    auto s1 = Shape<_2,Shape<_3,_4>>();
    auto s2 = Shape<_5,Shape<_6,_7>>();
    auto s3 = make_shape(s1,s2);
    
    Layout a_col = make_layout(s3, GenColMajor{});  //GenColMajor == LayoutLeft
    print_coalesce(a_col);

    Layout a_row = make_layout(s3, GenRowMajor{});  //GenRowMajor == LayoutRight
    print_coalesce(a_row);

    printf("\nCoalesce-Layout :");
    auto result = coalesce(a_col, Step<_1,_1>{});   //(_24,_210):(_1,_24) 

    auto b1 = coalesce(a_col,Step<_1,Step<_1,_1>>{});
    print(b1);
}
//H-Layout a :((_2,(_3,_4)),(_5,(_6,_7))):((_1,(_2,_6)),(_24,(_120,_720)))
//Coalesce-Layout :_5040:_1

/*
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
*/