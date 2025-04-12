#include <iostream>
#include <cstdlib> 
#include <chrono>
#include <cstring>

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

    // Time the addition
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Array size: " << N << " took " << elapsed.count() << " seconds.\n";

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
