#include "mma.h"
using namespace nvcuda;

__global__ void matmulT(float *C, half *A, half *B, int Ay, int Ax, int Bx)
{
    // warp rank in grid
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
    int cx = warp % (Bx / 16);    // (x,y) location if active tile
    int cy = warp / (Bx / 16);    // for current warp in C matrix
    int Atile_pos = cy * 16 * Bx; // start x (row) for first A tile
    int Btile_pos = cx * 16;      // start y (col) for first B tile

    // Declare the fragments as 16 x 16 tiles
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; // A
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; // B
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;              // C
    wmma::fill_fragment(c_frag, 0.0f);                                        // set C = 0

    // load A as 16x16 tile
    wmma::load_matrix_sync(a_frag, &A[Atile_pos], Ax);
    // load B as 16x16 tile
    wmma::load_matrix_sync(b_frag, &B[Btile_pos], Bx);
    // C = A*B + C
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
}
