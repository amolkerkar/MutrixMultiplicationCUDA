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

void run_parallel_mm_with_streams() {
    const int STREAMS = 8;
    float *A[STREAMS], *B[STREAMS], *C[STREAMS];
    cudaStream_t streams[STREAMS];

    size_t size = N * N * sizeof(float);

    for (int i = 0; i < STREAMS; i++) {
        cudaMallocManaged(&A[i], size);
        cudaMallocManaged(&B[i], size);
        cudaMallocManaged(&C[i], size);
        cudaStreamCreate(&streams[i]);

        // Initialize matrices
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

    // Launch matrix multiplications on separate CUDA streams
    for (int i = 0; i < STREAMS; i++) {
        matmul<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(A[i], B[i], C[i], N);
    }

    // Wait for all streams to complete
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Matrix Multiplication with %d Streams took: %f ms\n", STREAMS, elapsedTime);

    for (int i = 0; i < STREAMS; i++) {
        cudaFree(A[i]);
        cudaFree(B[i]);
        cudaFree(C[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    run_parallel_mm_with_streams();
    return 0;
}
