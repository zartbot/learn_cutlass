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

    //layout-a
    auto sa = make_shape(Int<4>{},Int<2>{},Int<3>{});
    auto da = make_stride(Int<2>{},Int<1>{},Int<8>{});
    auto a = make_layout(sa, da);
    Tensor ta =make_tensor(A, a);
    printf("\nLayout A: ");
    print(ta);

    //layout-a
    auto sb = make_shape(Int<4>{});
    auto db = make_stride(Int<2>{});
    auto b = make_layout(sb, db);
    Tensor tb =make_tensor(A, b);
    printf("\nLayout B: ");
    print_tensor(tb);

    auto b_star = complement(b, size(a));
    Tensor tb_star =make_tensor(A, b_star);
    printf("\nLayout B*: ");
    print_tensor(tb_star);

    auto c1 = composition(a,b);
    Tensor tc1 =make_tensor(A, c1);
    auto c2 = composition(a,b_star);
    Tensor tc2 =make_tensor(A, c2);
    printf("\nLayout A o B: ");
    print_tensor(tc1);
    printf("\nLayout A o B*: ");
    print_tensor(tc2);

    auto d = logical_divide(a,b);
    Tensor td =make_tensor(A, d);
     printf("\nLayout A div B: ");
    print_tensor(td);
  
}