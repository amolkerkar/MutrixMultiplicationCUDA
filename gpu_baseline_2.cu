#include <cuda_runtime.h>
#include <stdio.h>

#define N 2048

__global__ void matmul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void run_sequential_mm() {
    const int TASKS = 8;
    float *A[TASKS], *B[TASKS], *C[TASKS];
    size_t size = N * N * sizeof(float);

    for (int i = 0; i < TASKS; i++) {
        cudaMallocManaged(&A[i], size);
        cudaMallocManaged(&B[i], size);
        cudaMallocManaged(&C[i], size);
        for (int j = 0; j < N * N; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
        }
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < TASKS; i++) {
        matmul<<<numBlocks, threadsPerBlock>>>(A[i], B[i], C[i], N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Sequential Matrix Multiplication (%d tasks) took: %f ms\n", TASKS, elapsedTime);

    for (int i = 0; i < TASKS; i++) {
        cudaFree(A[i]);
        cudaFree(B[i]);
        cudaFree(C[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    run_sequential_mm();
    return 0;
}
