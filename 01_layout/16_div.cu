#include <getopt.h>
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

#define MAXN  128*128

int main()
{
    /*
   Layout raked_prod = Layout<Shape <Shape < _3,_2>,Shape <_4,_2>>,
                           Stride<Stride<_16,_1>,Stride<_4,_2>>>{};
Tile   subtile    = make_tile(Layout<_2,_3>{},    // Gather elements 2 : 3 from mode 0
                              Layout<_2,_4>{});   // Gather elements 2 : 4 from mode 1

print_layout(logical_divide(raked_prod, subtile));*/

// A: shape is (9,32)
auto layout_a = make_layout(make_shape (Int< 9>{}, make_shape (Int< 4>{}, Int<8>{})),
                            make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
// B: shape is (3,8)
auto tiler = make_tile(Layout<_3,_3>{},           // Apply     3:3     to mode-0
                       Layout<Shape <_2,_4>,      // Apply (2,4):(1,8) to mode-1
                              Stride<_1,_8>>{});

// ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))
auto ld = logical_divide(layout_a, tiler);
// ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
auto zd = zipped_divide(layout_a, tiler);

print_layout(ld);
printf("\n zip-div :\n");
print_layout(zd);
}