#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 1000000
#define blocksize 1024
__global__ void hello(int* d_A,int*d_B,int*d_C){
 int idx = threadIdx.x;
 d_C[idx] = d_A[idx] + d_B[idx];
}

int main(){

int* h_A = (int*)malloc(sizeof(int)*N);
int* h_B = (int*)malloc(sizeof(int)*N);
int* h_C = (int*)malloc(sizeof(int)*N);

int* d_A;
int* d_B;
int* d_C;


for(int i=0;i<10;i++){
    h_A[i] = rand()%100;
    h_B[i] = rand()%100;
}
cudaMalloc((void**)&d_A,sizeof(int)*N);
cudaMemcpy(d_A,h_A,sizeof(int)*N,cudaMemcpyHostToDevice);

cudaMalloc((void**)&d_B,sizeof(int)*N);
cudaMemcpy(d_B,h_B,sizeof(int)*N,cudaMemcpyHostToDevice);
cudaMalloc((void**)&d_C,sizeof(int)*N);

hello<<<1,N>>>(d_A,d_B,d_C);

cudaDeviceSynchronize();

cudaMemcpy(h_C,d_C,N*sizeof(int),cudaMemcpyDeviceToHost);

for(int i=0;i<N;i++)
printf("\n %d %d %d", h_A[i],h_B[i],h_C[i]);

return 0;

}