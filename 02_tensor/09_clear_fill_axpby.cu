#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
  
    Tensor A = make_tensor<int>(make_shape(_4{},_8{}),GenColMajor{});
    fill(A, 7);
    print_tensor(A);
    clear(A);
    print_tensor(A);
    fill(A, 3);

    Tensor B = make_tensor<int>(make_shape(_4{},_8{}),GenColMajor{});
    fill(B, 2);

    //B = 3 * A + 2 * B
    axpby(3,A, 2, B);
    print_tensor(B);
}

