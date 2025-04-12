#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "timer.h"

// CUDA kernel for array addition
__global__ void addArrays(float* A, float* B, float* C, size_t N) {
    int num_threads_total = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int num_elements_computed = 0;
    int num_elements_to_compute = N / num_threads_total;
    if (N % num_threads_total != 0) {
        num_elements_to_compute++;
    }

    while (num_elements_computed < num_elements_to_compute) {
        int cur_index = tid + num_threads_total * num_elements_computed;
        if (cur_index < N) {
            C[cur_index] = A[cur_index] + B[cur_index];
        }
        num_elements_computed++;
    }
}

bool verifyResults(float* A, float* B, float* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        if (C[i] != A[i] + B[i]) {
            std::cerr << "Mismatch at index " << i << ": C[" << i << "] = "
                      << C[i] << " but expected " << A[i] + B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K>\n";
        return 1;
    }

    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cerr << "K must be a positive integer.\n";
        return 1;
    }

    size_t N = static_cast<size_t>(K) * 1000000;

    // Unified memory allocation
    float *A, *B, *C;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    // Initialize A and B
    for (size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    // Scenario 1: 1 block, 1 thread
    initialize_timer();
    start_timer();
    addArrays<<<1, 1>>>(A, B, C, N);
    cudaDeviceSynchronize();
    stop_timer();
    std::cout << "1 block, 1 thread per block: " << elapsed_time() << " seconds.\n";

    if (verifyResults(A, B, C, N)) {
        std::cout << "Array addition is correct!\n";
    } else {
        std::cerr << "Array addition failed!\n";
    }

    // Scenario 2: 1 block, 256 threads
    int threadsPerBlock = 256;
    int numBlocks = 1;

    initialize_timer();
    start_timer();
    addArrays<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
    stop_timer();
    std::cout << "1 block, 256 threads per block: " << elapsed_time() << " seconds.\n";

    if (verifyResults(A, B, C, N)) {
        std::cout << "Array addition is correct!\n";
    } else {
        std::cerr << "Array addition failed!\n";
    }

    // Scenario 3: Multiple blocks, 256 threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    initialize_timer();
    start_timer();
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
    stop_timer();
    std::cout << blocksPerGrid << " blocks, 256 threads per block: " << elapsed_time() << " seconds.\n";

    if (verifyResults(A, B, C, N)) {
        std::cout << "Array addition is correct!\n";
    } else {
        std::cerr << "Array addition failed!\n";
    }

    // Free Unified Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
