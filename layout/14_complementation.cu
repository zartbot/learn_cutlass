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

    auto sa = make_shape(Int<2>{},Int<2>{});
    auto a = make_layout(sa, Stride<_1,_6>{});
    Tensor ta =make_tensor(A, a);
    print_tensor(ta);

    auto c = complement(a, 24);
    Tensor tc =make_tensor(A, c);
    print_tensor(tc);
   
}