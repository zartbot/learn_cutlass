#include <getopt.h>
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

using namespace cute;

#define MAXN  128*128

int main()
{
    // initial memory with physical layout
    int* A = (int*)malloc(MAXN * sizeof(int));
    for(int i =0 ; i < MAXN ; i++){
	    A[i]=int(i);
    }   

    auto sa = make_shape(Int<20>{});
    auto a = make_layout(sa, Stride<_2>{});
    Tensor ta =make_tensor(A, a);
    print_tensor(ta);

    auto sb = make_shape(Int<5>{}, Int<4>{});
    auto b = make_layout(sb, make_stride(Int<4>{}, Int<2>{}));
    Tensor tb =make_tensor(A, b);
    print_tensor(tb);
    print(cosize(b));
    
    auto c = composition(a, b);
    Tensor tc =make_tensor(A, c);
    print_tensor(tc);
   
}