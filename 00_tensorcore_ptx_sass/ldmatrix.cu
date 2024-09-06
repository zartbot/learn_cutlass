#include <stdio.h>
#include <stdint.h>
#include "cuda_fp16.h"
// #include "mma.h"

#define WARP_SIZE 32

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define LDMATRIX_X1T(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2T(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4T(R0, R1, R2, R3, addr)                                                  \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                   \
                 : "r"(addr))

__global__ void TestLDMatrix(void)
{
    const int tid = threadIdx.x;

    __shared__ uint16_t M[4 * 16 * 16];
    if (tid == 0)
    {
        int offset = 0;
        for (int i = 0; i < 4; ++i){
            for (int j = 0; j < 16; ++j){
                for (int k = 0; k < 16; ++k)
                {
                    M[offset] = static_cast<uint16_t>((i+1) * 10000 + (j+1) * 100 + k+1);
                    printf(" %6d",M[offset]);
                    offset++;
                }
                printf("\n");
            }
             printf("\n");
        }
    }

    __syncthreads();

    int offset = tid * 32;

    uint32_t addr = __cvta_generic_to_shared(M + offset);

    uint32_t frag[4];
    //LDMATRIX_X1(frag[0],addr);
    LDMATRIX_X4T(frag[0], frag[1], frag[2], frag[3], addr);
    uint16_t data[4][2];
    for (int i = 0; i < 4; ++i)
    {
        data[i][0] = static_cast<uint16_t>(frag[i] & 0xFFFF);
        data[i][1] = static_cast<uint16_t>((frag[i] >> 16) & 0xFFFF);
    }
    printf("OFFSET %4d  tid: %3d | A | %6d %6d | %6d %6d | %6d %6d | %6d %6d |\n", offset, tid,
           int(data[0][0]), int(data[0][1]), int(data[1][0]), int(data[1][1]),
           int(data[2][0]), int(data[2][1]), int(data[3][0]), int(data[3][1]));
}

int main(void)
{
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1);
    TestLDMatrix<<<gridDim, blockDim>>>();

    cudaDeviceReset();

    return 0;
}


        /*
    int aTile_index = tid % 16 * 8 + tid / 16 * 8;

        const uint32_t address =
            cvta_to_shared_u32(M) + sizeof(uint16_t) * ((tid%8) * 8);*/

