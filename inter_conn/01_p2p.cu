/*%****************************************************************************80
%  Code: 
%   ncclSendRecv.cu
%
%  Purpose:
%   Implements sample send/recv code using the package NCCL (p2p).
%
%  Modified:
%   Aug 18 2020 10:57 
%
%  Author:
%   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
%
%  How to Compile:
%   nvcc ncclSendRecv.cu -o object -lnccl  
%
%  HowtoExecute: 
%   ./object 
%                         
%****************************************************************************80*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nccl.h>

__global__ void kernel(int *a, int rank) { 

  if(rank == 0)
    printf("%d\t", a[threadIdx.x]); 
      else
        printf("%d\t", a[threadIdx.x]*10); 
}
 
void show_all(int *in, int n){

 printf("\n");

 for(int i=0; i < n; i++)
    printf("%d\t", in[i]);
      
 printf("\n");

}/*show_all*/


int main(int argc, char* argv[]) {

  int size = 4096;

  /*Get current amounts number of GPU*/
  int nGPUs = 0;
  cudaGetDeviceCount(&nGPUs);
  printf("nGPUs = %d\n",nGPUs);

  /*List GPU Device*/
  int *DeviceList = (int *) malloc ( nGPUs * sizeof(int));

  for(int i = 0; i < nGPUs; ++i)
      DeviceList[i] = i;
  
  /*NCCL Init*/
  ncclComm_t* comms         = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s           = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList); 

  /*General variables*/
  int *host       = (int*) malloc(size      * sizeof(int));
  int **sendbuff  = (int**)malloc(nGPUs     * sizeof(int*));
  int **recvbuff  = (int**)malloc(nGPUs     * sizeof(int*));
  
  /*Population of vector*/
  for(int i = 0; i < size; i++)
      host[i] = i + 1;

  show_all(host, size);

  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      cudaStreamCreate(&s[g]);
      cudaMalloc(&sendbuff[g], size * sizeof(int));
      cudaMalloc(&recvbuff[g], size * sizeof(int));
     
      if(g == 0)
        cudaMemcpy(sendbuff[g], host, size * sizeof(int),cudaMemcpyHostToDevice);
       
  }/*for*/
  
  ncclGroupStart();        
    for (int i = 0 ; i < 10240000 ; ++i ) {

  	for(int g = 0; g < nGPUs; g++) {
          ncclSend(sendbuff[0], size, ncclInt, g, comms[g], s[g]);
    	    ncclRecv(recvbuff[g], size, ncclInt, g, comms[g], s[g]);
    }
    }
  
  ncclGroupEnd();          


  for(int g = 0; g < nGPUs; g++) {
     ncclCommDestroy(comms[g]);
  }
  
  free(s);
  free(host);
  
  cudaFree(sendbuff);
  cudaFree(recvbuff);

  return 0;

}


/*

  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      printf("\nThis is device %d\n", g);
      if(g==0)
      kernel <<< 1 , size >>> (sendbuff[g], g); 
      else
      kernel <<< 1 , size >>> (recvbuff[g], g); 
      cudaDeviceSynchronize();
  }

 printf("\n");

  for (int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      cudaStreamSynchronize(s[g]);
  }

  
  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      cudaStreamDestroy(s[g]);
  }

*/