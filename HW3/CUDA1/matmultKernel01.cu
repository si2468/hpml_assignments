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
#include <stdio.h>

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A_mat, Matrix B_mat, Matrix C_mat){
    
    //C_mat.elements[100] = 3.14f;    

  // matrix blocks
  float *A, *B, *C;
  A = A_mat.elements;
  B = B_mat.elements;
  C = C_mat.elements;
  
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  //Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + BLOCK_SIZE * block_col];

  // Each thread computes four elements of Csub in its copy of CValues
  float Cvalue00 = 0;
  float Cvalue01 = 0;
  float Cvalue10 = 0;
  float Cvalue11 = 0;


  // row and column indices within a tile for this thread
  int r0 = thread_row * 2;
  int r1 = r0 + 1;
  int c0 = thread_col * 2;
  int c1 = c0 + 1;

  __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
  __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

  int elements_per_thread = FOOTPRINT_SIZE * FOOTPRINT_SIZE / (BLOCK_SIZE * BLOCK_SIZE);


  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A_mat.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    //Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m]; // tile (block_row, m)
    //Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col]; // tile (m, block_col)

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B

    // each thread will load a 2 x 2 submatrix from global memory into shared memory

    // current tile start points for shared matrix A
    int tile_origin_row_A = block_row * FOOTPRINT_SIZE;
    int tile_origin_col_A = m * FOOTPRINT_SIZE;

    for (int i = 0; i < elements_per_thread / 2; ++i) {
        for (int j = 0; j < elements_per_thread / 2; ++j) {
            // Each thread loads 4 elements into shared memory

            // sh row and sh col are the indices in a 32 x 32 tile -> this also corresponds to the 32 x 32 tiles in global memory. 
            // Just have to find which tile it is
            int sh_row = thread_row * elements_per_thread / 2 + i;
            int sh_col = thread_col * elements_per_thread / 2 + j;

            int global_row = tile_origin_row_A + sh_row;
            int global_col = tile_origin_col_A + sh_col;

            shared_A[sh_row][sh_col] = A[global_row * A_mat.stride + global_col];
        }
    }
    __syncthreads();

    // current tile start points for shared matrix B
    int tile_origin_row_B = m * FOOTPRINT_SIZE;
    int tile_origin_col_B = block_col * FOOTPRINT_SIZE;
    
    for (int i = 0; i < elements_per_thread / 2; ++i) {
        for (int j = 0; j < elements_per_thread / 2; ++j) {
            int sh_row = thread_row * elements_per_thread / 2 + i;
            int sh_col = thread_col * elements_per_thread / 2 + j;
    
            int global_row = tile_origin_row_B + sh_row;
            int global_col = tile_origin_col_B + sh_col;
    
            shared_B[sh_row][sh_col] = B[global_row * B_mat.stride + global_col];
        }
    }
    
    // Synchronize to ensure all elements are read
    __syncthreads();

    // Perform the multiplication of one row of A with one column of B
    // This is the dot product between the row from A and column from B

    #pragma unroll
    for (int k = 0; k < FOOTPRINT_SIZE; ++k) {
        float A_r0_k = shared_A[r0][k];
        float A_r1_k = shared_A[r1][k];
        float B_k_c0 = shared_B[k][c0];
        float B_k_c1 = shared_B[k][c1];

        Cvalue00 += A_r0_k * B_k_c0;
        Cvalue01 += A_r0_k * B_k_c1;
        Cvalue10 += A_r1_k * B_k_c0;
        Cvalue11 += A_r1_k * B_k_c1;
    }
    // Synchronize threads to ensure all Cvalues are computed before loading the next block
    __syncthreads();
    }

    // Write the computed Cvalue to the output matrix C at the appropriate location
    __syncthreads();

    int global_row = block_row * FOOTPRINT_SIZE + r0;
    int global_col = block_col * FOOTPRINT_SIZE + c0;

    C[(global_row + 0) * C_mat.stride + (global_col + 0)] = Cvalue00;
    C[(global_row + 0) * C_mat.stride + (global_col + 1)] = Cvalue01;
    C[(global_row + 1) * C_mat.stride + (global_col + 0)] = Cvalue10;
    C[(global_row + 1) * C_mat.stride + (global_col + 1)] = Cvalue11;
}
