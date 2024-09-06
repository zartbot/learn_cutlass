#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
using namespace std;
__global__ void test_wmma()
{
        // Warps within a block read from 256 byte aligned strided adresses to avoid
        // bank conflicts (makes no difference).
        __shared__ __half smem[1024 * 8];
        __half* A = smem + threadIdx.y * 1024 + threadIdx.y * 16;
        __half* B = smem + threadIdx.y * 1024 + threadIdx.y * 16 + 256;
        __half* C = smem + threadIdx.y * 1024 + threadIdx.y * 16 + 512;

        // Matrix A is read once, and accumulator is filled once.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;
        wmma::fill_fragment( acc_frag, 0.0f );
        wmma::load_matrix_sync( a_frag, A, 16 );

#pragma unroll
        for ( int i = 0; i < 20; i++ )
        {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync( b_frag, B, 16 );
                wmma::mma_sync( acc_frag, a_frag, b_frag, acc_frag );
        }

        wmma::store_matrix_sync( C, acc_frag, 16, wmma::mem_col_major );
}
void TestWMMA()
{
        int threads = 256;
        int blocks = 10000;
        test_wmma<<<blocks, threads>>>();
}
int main(){

        TestWMMA();
        cudaDeviceSynchronize();
        TestWMMA();
        cudaDeviceSynchronize();
}