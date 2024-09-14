#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include "common.hpp"
#include "cublas_v2.h"

void launch_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                 CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__global__ void SimpleKernel(float *local_mem, float *remote_mem1, float *remote_mem2, float *remote_mem3)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 1024768; ++j)
    {
        for (int i = 0; i < 1000; ++i)
        {
            remote_mem1[idx + i * 32] = local_mem[idx + i * 32] * 2.0f;
            remote_mem2[idx + i * 32] = local_mem[idx + i * 32] * 2.0f;
            remote_mem3[idx + i * 32] = local_mem[idx + i * 32] * 2.0f;
            // __nanosleep(1000);
        }
    }
}

__global__ void SmallKernel(float *local_mem, float *remote_mem1, float *remote_mem2, float *remote_mem3)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; ++i)
    {
        remote_mem1[idx + i * 32] = local_mem[idx + i * 32] * 2.0f;
        remote_mem2[idx + i * 32] = local_mem[idx + i * 32] * 2.0f;
        remote_mem3[idx + i * 32] = local_mem[idx + i * 32] * 2.0f;
    }
    __nanosleep(1000000);
}

int main()
{
    uint32_t size = pow(2, 30); // Memory Copy Size
    const int ngpu = 4;

    float *dev[ngpu];
    for (int i = 0; i < ngpu; ++i)
    {
        cudaSetDevice(0);
        cudaMalloc((void **)&dev[i], size);
        for (int j = 0; j < ngpu; ++j)
        {
            if (i != j)
            {
                cudaDeviceEnablePeerAccess(i, j);
            }
        }
    }
    cudaSetDevice(0);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M_GLOBAL * K_GLOBAL * sizeof(half));
    cudaMalloc(&d_b, K_GLOBAL * N_GLOBAL * sizeof(half));
    cudaMalloc(&d_c, M_GLOBAL * N_GLOBAL * sizeof(half));

    const int nStreams = 4;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < ITER; i++)
    {
        launch_gemm(M_GLOBAL, N_GLOBAL, K_GLOBAL, d_a, d_b, d_c, alpha, beta);
        // SimpleKernel<<<1, 32>>>(dev[0], dev[1], dev[2], dev[3]); // 执行GPU0 Kernel
        SmallKernel<<<1, 32, 0, stream[2]>>>(dev[0], dev[1], dev[2], dev[3]); 
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    long workload = long(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2 * ITER;
    double avg_Gflops = ((double)workload / 1e9) / (double(msec) / 1e3);
    printf("Average Performance  %10.1lf Gflops\n", avg_Gflops);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamDestroy(stream[i]);
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < ngpu; ++i)
    {
        cudaFree(dev[i]);
    }
}