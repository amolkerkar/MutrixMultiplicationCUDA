# CUDA Streams: Matrix Multiplication Concurrency Benchmark

This project benchmarks the performance of **independent matrix multiplications** using:

- **Baseline sequential GPU execution**
- **CUDA Streams for concurrent execution**

It demonstrates how concurrent kernel execution can better utilize GPU hardware, by mapping multiple independent ML-like tasks onto different **CUDA streams**.

## Project Structure

| File                      | Description                                    |
|---------------------------|------------------------------------------------|
| `gpu_baseline.cu`         | Baseline version: 3 matrix multiplications (1024×1024), one after another on the default stream |
| `gpu_streamed.cu`         | Streamed version: 3 matrix multiplications using 3 CUDA streams |
| `gpu_baseline_2.cu`       | Baseline version: 8 matrix multiplications (2048×2048), run sequentially |
| `gpu_cudastreams_2.cu`    | Streamed version: 8 matrix multiplications using 8 CUDA streams (parallel execution) |

## Hardware

- **GPU**: NVIDIA RTX 4060 Laptop GPU
- **CUDA Version**: 12.8
- **OS**: Windows


## CUDA Concepts Used

- **Streams**: To allow overlapping of independent GPU tasks.
- **Unified Memory (`cudaMallocManaged`)**: Simplifies memory management.
- **CUDA Events**: For accurate GPU timing.
- **Nvidia Nsight Systems**: For visual GPU stream profiling and timeline analysis.

## 📈 Performance Comparison

| File                         | Matrix Size | Streams | Time (ms) | Speedup      |
|------------------------------|-------------|---------|-----------|--------------|
| `gpu_baseline.cu`            | 1024×1024   | 1       | ~7.5 ms   | Baseline     |
| `gpu_streamed.cu`            | 1024×1024   | 3       | ~6.2 ms   | ~17% faster  |
| `gpu_baseline_2.cu`          | 2048×2048   | 1       | ~140 ms   | Baseline     |
| `gpu_cudastreams_2.cu`       | 2048×2048   | 8       | ~128 ms   | ~8–10% faster|

Note: The speedup is bounded by SM availability (8 on my device) and memory bandwidth. Larger matrices and more compute-heavy tasks reveal more benefit.


## Nsys profiling for Baseline vs CUDA streamed approach:

### 🔹 Baseline (8MM Sequential)
![BaseLine MM](https://github.com/user-attachments/assets/8c2788dd-266b-4825-b18f-46d7d58f0e92)

### 🔸 CUDA Streams (8MM Parallel)
![CUDAStreamed MM](https://github.com/user-attachments/assets/cf62eea8-2923-4247-a14a-002411755f88)

- Multiple streams show clear **overlap of kernel execution**.
- SMs are more **fully utilized** in the streamed version.
- Verified using **NVIDIA Nsight Systems**.


## 📌 How to Run

### 1. Compile
```bash
nvcc -o gpu_baseline gpu_baseline.cu
nvcc -o gpu_streamed gpu_streamed.cu
nvcc -o gpu_baseline_2 gpu_baseline_2.cu
nvcc -o gpu_cudastreams_2 gpu_cudastreams_2.cu
