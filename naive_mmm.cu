#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define M 3
#define N 3
#define P 3

// CPU version of matrix multiplication
void MMM_CPU(float* A, float* B, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// CUDA kernel for naive matrix multiplication
__global__ void MMM_GPU(float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

int main() {
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * P * sizeof(float);
    size_t size_C = M * P * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_CPU = (float*)malloc(size_C);
    float *h_C_GPU = (float*)malloc(size_C);

    // Initialize matrices A and B
    for (int i = 0; i < M * N; i++) h_A[i] = 1.0f;
    for (int i = 0; i < N * P; i++) h_B[i] = 2.0f;

    // --- CPU computation ---
    clock_t cpu_start = clock();
    MMM_CPU(h_A, h_B, h_C_CPU);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU Time: %f ms\n", cpu_time);

    // --- GPU computation ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    MMM_GPU<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %f ms\n", gpu_time);

    cudaMemcpy(h_C_GPU, d_C, size_C, cudaMemcpyDeviceToHost);

    // --- Verify correctness ---
    int errors = 0;
    for (int i = 0; i < M * P; i++) {
        if (abs(h_C_CPU[i] - h_C_GPU[i]) > 1e-3) {
            errors++;
            if (errors < 10)
                printf("Mismatch at %d: CPU=%f GPU=%f\n", i, h_C_CPU[i], h_C_GPU[i]);
        }
    }
    if (errors == 0) printf("Results match.\n");
    else printf("Results do not match! Errors: %d\n", errors);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);

    return 0;
}
