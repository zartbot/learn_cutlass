#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Macro for checking cuda errors following a cuda launch or api call
#define CUDA_CHECK(ans)                       \
   {                                         \
       gpuAssert((ans), __FILE__, __LINE__); \
   }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
       fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file,
               line);
       exit(code);
   }
}

// Macro to report cuda errors following a cuda launch or api call
#define CUDA_REPORT(ans)                       \
   {                                         \
       gpuReport((ans), __FILE__, __LINE__); \
   }

inline void gpuReport(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
       fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file,
               line);
   }
}