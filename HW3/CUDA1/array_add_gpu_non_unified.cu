#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include "timer.h"

// CUDA kernel for array addition
__global__ void addArrays(float* A, float* B, float* C, size_t N) {
    int num_threads_total = blockDim.x * gridDim.x;

    int num_elements_computed = 0;
    int num_elements_to_compute = N / num_threads_total;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N % num_threads_total != 0){
        num_elements_to_compute++;
    }
    while(num_elements_computed < num_elements_to_compute){
        int cur_index = tid + num_threads_total * num_elements_computed;
        if (cur_index < N){
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

    // Host memory allocation
    float* A = (float*) malloc(N * sizeof(float));
    float* B = (float*) malloc(N * sizeof(float));
    float* C = (float*) malloc(N * sizeof(float));

    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed\n";
        free(A); free(B); free(C);
        return 1;
    }

    // Initialize A and B
    for (size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    // Device memory allocation
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Time the addition (for all three scenarios)

    // Scenario 1: 1 block, 1 thread
    initialize_timer();
    start_timer();
    addArrays<<<1, 1>>>(d_A, d_B, d_C, N); // Each call adds one element
    cudaDeviceSynchronize();
    stop_timer();
    std::cout << "Scenario 1 (1 block, 1 thread): " << elapsed_time() << " seconds.\n";

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);


    // Verify results
    if (verifyResults(A, B, C, N)) {
        std::cout << "Array addition is correct!\n";
    } else {
        std::cerr << "Array addition failed!\n";
    }

    // Scenario 2: 1 block, 256 threads
    int threadsPerBlock = 256;
    int numBlocks = 1 ;

    initialize_timer();
    start_timer();
    addArrays<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    stop_timer();
    std::cout << "Scenario 2 (1 block, 256 threads): " << elapsed_time() << " seconds.\n";

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    if (verifyResults(A, B, C, N)) {
        std::cout << "Array addition is correct!\n";
    } else {
        std::cerr << "Array addition failed!\n";
    }

    // Scenario 3: Multiple blocks, 256 threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    initialize_timer();
    start_timer();
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    stop_timer();
    std::cout << "Scenario 3 (Multiple blocks, 256 threads per block): " << elapsed_time() << " seconds.\n";

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    if (verifyResults(A, B, C, N)) {
        std::cout << "Array addition is correct!\n";
    } else {
        std::cerr << "Array addition failed!\n";
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
