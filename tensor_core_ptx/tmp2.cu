#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void test_wmma(half  *C, half *A, half *B)
{
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

//        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a_frag;
//        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        

        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;

        wmma::load_matrix_sync( a_frag, A, 16 );
        wmma::load_matrix_sync( b_frag, B, 16 );
        wmma::fill_fragment( acc_frag, 0.0f );
        
        wmma::mma_sync( acc_frag, a_frag, b_frag, acc_frag );
        wmma::store_matrix_sync( C, acc_frag, 16, wmma::mem_row_major );
}

//  nvcc -c -arch sm_70 --ptx  tmp2.cu 
// nvcc -c -arch sm_70 tmp2.cu ; cuobjdump -sass tmp2.o > tmp.sass

