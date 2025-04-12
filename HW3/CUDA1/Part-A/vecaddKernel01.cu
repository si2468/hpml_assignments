///
/// vecAddKernel01.cu
/// Adding vectors with memory coelescing

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int num_threads = blockDim.x * gridDim.x;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int num_elements_computed_per_thread = 0;

    while(num_elements_computed_per_thread < N){
        int cur_index = tid + num_elements_computed_per_thread * num_threads;
        C[cur_index] = A[cur_index] + B[cur_index];
        num_elements_computed_per_thread++;
    }
}