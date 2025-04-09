#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

__global__ void ml_task(float* a, float* b, float* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = a[idx] * b[idx];
}

void run_parallel_tasks_with_streams() {
    float *a[3], *b[3], *c[3];
    cudaStream_t streams[3];

    for (int i = 0; i < 3; i++) {
        cudaMallocManaged(&a[i], N * sizeof(float));
        cudaMallocManaged(&b[i], N * sizeof(float));
        cudaMallocManaged(&c[i], N * sizeof(float));
        for (int j = 0; j < N; j++) { a[i][j] = 2.0f; b[i][j] = 3.0f; }
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 3; i++) {
        ml_task<<<(N + 255)/256, 256, 0, streams[i]>>>(a[i], b[i], c[i]);
    }

    for (int i = 0; i < 3; i++) cudaStreamSynchronize(streams[i]);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Streamed GPU time: %f ms\n", ms);

    for (int i = 0; i < 3; i++) {
        cudaFree(a[i]); cudaFree(b[i]); cudaFree(c[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    run_parallel_tasks_with_streams();
    return 0;
}
