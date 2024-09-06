#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

#define MAXN 128*128
#define PRINTTENSOR(name,  tensor) \
    print(name);                          \
    print("\nTensor : ");                 \
    print_tensor(tensor);                 \
    print("\n");                    

int main()
{

    // initial memory with physical layout
    int* A = (int*)malloc(MAXN * sizeof(int));
    for(int i =0 ; i < MAXN ; i++){
	    A[i]=int(i);
    }   

    auto shape_1d = make_shape(Int<8>{});

    //Layout _8:_1
    Tensor t_1d = make_tensor(A, make_layout(shape_1d, make_stride(_1{})));
    PRINTTENSOR("1d layout",t_1d)
    
    //Layout _8:_2
    Tensor t_s2 = make_tensor(A,make_layout(shape_1d, make_stride(_2{})));
    PRINTTENSOR("1d stride2",t_s2)
    
    //Layout _8:_m1
    Tensor t_s_m1 = make_tensor(A+7,make_layout(shape_1d, make_stride(_m1{})));
    PRINTTENSOR("1d stride -1",t_s_m1)

    //Layout _8:_m2
    Tensor t_s_m2 = make_tensor(A+16,make_layout(shape_1d, make_stride(_m2{})));
    PRINTTENSOR("1d stride -1",t_s_m2)

}

/*


    printf("Coord : ");
    for (int i = 0 ; i < shape<0>(layout_1d); ++i) {
        printf("%3d ", i);
    }
    printf("\n");
    printf("Index : ");
    for (int i = 0 ; i < shape<0>(layout_1d); ++i) {
        printf("%3d ", A[i]);
    }
    printf("\n");           // 1D tensor
    auto shape_1d = make_shape(Int<8>{});
    auto stride_1d = make_stride(Int<-1>{});
    auto layout_1d = make_layout(shape_1d, stride_1d);
    Tensor t_1d = make_tensor(A+7, layout_1d);
    PRINTTENSOR("1d layout",layout_1d,t_1d)
*/