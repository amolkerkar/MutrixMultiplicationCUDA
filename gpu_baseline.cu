#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

__global__ void ml_task(float* a, float* b, float* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = a[idx] * b[idx];
}

void run_sequential_tasks() {
    float *a, *b, *c;
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&c, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = 2.0f; b[i] = 3.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 3; i++)
    {
        ml_task<<<(N + 255)/256, 256>>>(a, b, c);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Bseline GPU time: %f ms\n", ms);

    cudaFree(a); cudaFree(b); cudaFree(c);
}

int main() {
    run_sequential_tasks();
    return 0;
}
