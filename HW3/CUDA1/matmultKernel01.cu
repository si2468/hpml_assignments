///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // Matrix blocks
  float *Asub, *Bsub, *Csub;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub-matrix Csub of C
  // Each THREAD computes a 2x2 block (4 values)
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  // Each thread computes four values of Csub in its copy of Cvalue
  float Cvalue[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Array to store the four results

  // Loop over all sub-matrices in block_row of A and block_col of B
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
    Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

    // Declare shared memory for A and B
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread loads one element of ASub and one element of Bsub into shared memory
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Compute four values of Cvalue (2x2 block) by performing the dot product
    #pragma unroll
    for(int e = 0; e < BLOCK_SIZE; ++e) {
        // Each thread computes values for a 2x2 block of C
        Cvalue[0] += shared_A[thread_row][e] * shared_B[e][thread_col];        // C[2*thread_row, 2*thread_col]
        Cvalue[1] += shared_A[thread_row][e] * shared_B[e][thread_col + 1];    // C[2*thread_row, 2*thread_col + 1]
        Cvalue[2] += shared_A[thread_row + 1][e] * shared_B[e][thread_col];    // C[2*thread_row + 1, 2*thread_col]
        Cvalue[3] += shared_A[thread_row + 1][e] * shared_B[e][thread_col + 1];// C[2*thread_row + 1, 2*thread_col + 1]
    }

    // Synchronize to ensure all Cvalues are computed before the next iteration
    __syncthreads();
  }

  // Write Csub to GLOBAL memory. Each thread writes its four computed values to the output matrix C.
  // Writing to 2x2 submatrix
  Csub[2 * thread_row * C.stride + 2 * thread_col] = Cvalue[0];        // C[2*thread_row, 2*thread_col]
  Csub[2 * thread_row * C.stride + 2 * thread_col + 1] = Cvalue[1];    // C[2*thread_row, 2*thread_col + 1]
  Csub[(2 * thread_row + 1) * C.stride + 2 * thread_col] = Cvalue[2];  // C[2*thread_row + 1, 2*thread_col]
  Csub[(2 * thread_row + 1) * C.stride + 2 * thread_col + 1] = Cvalue[3]; // C[2*thread_row + 1, 2*thread_col + 1]
}




