#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <iostream>

// various dimensions and definitions
#define INPUT_CHANNELS 3
#define INPUT_HEIGHT 1024
#define INPUT_WIDTH 1024
#define NUM_FILTERS 64
#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3
#define PADDING 1
#define STRIDE 1
#define DILATION 1

// tile dimensions
#define TILE_WIDTH 16
// account for padding
#define SHARED_TILE_WIDTH (TILE_WIDTH + FILTER_WIDTH - 1)

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// naive convolution Kernel
__global__ void conv_naive_kernel(double* input_padded,
                                  double* filter_weights,
                                  double* output_tensor,
                                  int input_channels, int input_height, int input_width,
                                  int num_filters, int filter_height, int filter_width) {
                                    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_idx = blockIdx.z; 

    double result = 0.0;

    for (int channel = 0; channel < input_channels; ++channel) {
        for (int fh = 0; fh < filter_height; ++fh) {
            for (int fw = 0; fw < filter_width; ++fw) {
                int input_x = x + fw;
                int input_y = y + fh;

                int flipped_fw = filter_width - 1 - fw;
                int flipped_fh = filter_height - 1 - fh;

                int filter_index = ((filter_idx * input_channels + channel) * filter_height + flipped_fh) * filter_width;
                filter_index +=flipped_fw;
                int input_index = (channel * (input_height + 2 * PADDING) + input_y) * (input_width + 2 * PADDING);
                input_index += input_x;

                result += filter_weights[filter_index] * input_padded[input_index];
            }
        }
    }

    int output_index = (filter_idx * input_height + y) * input_width + x;
    output_tensor[output_index] = result;
}

// tiled convolution Kernel
__global__ void conv_tiled_kernel(double* input_padded,
                                  double* filter_weights,
                                  double* output_tensor,
                                  int input_channels, int input_height, int input_width,
                                  int num_filters, int filter_height, int filter_width) {

    __shared__ double tile[INPUT_CHANNELS][SHARED_TILE_WIDTH][SHARED_TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int filter_idx = blockIdx.z;

    double result = 0.0;

    for (int c = 0; c < input_channels; ++c) {

        // include padding around shared tile
        for (int i = ty; i < SHARED_TILE_WIDTH; i += TILE_WIDTH) {
            for (int j = tx; j < SHARED_TILE_WIDTH; j += TILE_WIDTH) {
                int input_row = blockIdx.y * TILE_WIDTH + i;
                int input_col = blockIdx.x * TILE_WIDTH + j;
                
                int height_boundary = input_height + 2 * PADDING;
                int width_boundary = input_width + 2 * PADDING;
                if (input_row < height_boundary && input_col < width_boundary) {
                    tile[c][i][j] = input_padded[(c * height_boundary + input_row) * width_boundary + input_col];
                } else {
                    tile[c][i][j] = 0.0;
                }
            }
        }
        __syncthreads();

        for (int i = 0; i < filter_height; ++i) {
            for (int j = 0; j < filter_width; ++j) {
                int flipped_i = filter_height - 1 - i;
                int flipped_j = filter_width - 1 - j;
                int filter_index = ((filter_idx * input_channels + c) * filter_height + flipped_i) * filter_width + flipped_j;
                result += filter_weights[filter_index] * tile[c][ty + i][tx + j];
            }
        }
        __syncthreads();
    }
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int output_index = (filter_idx * input_height + row_o) * input_width + col_o;
    output_tensor[output_index] = result;
}

int main() { 
    // we need to account for padding here
    size_t input_padded_bytes = INPUT_CHANNELS * (INPUT_HEIGHT + 2 * PADDING) * (INPUT_WIDTH + 2 * PADDING) * sizeof(double);
    size_t filter_bytes = NUM_FILTERS * INPUT_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH * sizeof(double);
    size_t output_bytes = NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double);

    // Host allocations with padding
    double* host_input_padded = (double*)malloc(input_padded_bytes);
    double* host_filters = (double*)malloc(filter_bytes);
    double* host_output = (double*)malloc(output_bytes);

    // I[c, x, y] = c * (x + y)
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int y = 0; y < INPUT_HEIGHT; ++y) {
            for (int x = 0; x < INPUT_WIDTH; ++x) {
                int padded_index = (c * (INPUT_HEIGHT + 2 * PADDING) + (y + PADDING)) * (INPUT_WIDTH + 2 * PADDING) + (x + PADDING);
                host_input_padded[padded_index] = c * (x + y);
            }
        }
    }

    // generate the filters according to F[k, c, i, j] = (c + k) * (i + j)
    for (int k = 0; k < NUM_FILTERS; ++k) {
        for (int c = 0; c < INPUT_CHANNELS; ++c) {
            for (int fh = 0; fh < FILTER_HEIGHT; ++fh) {
                for (int fw = 0; fw < FILTER_WIDTH; ++fw) {
                    int index = ((k * INPUT_CHANNELS + c) * FILTER_HEIGHT + fh) * FILTER_WIDTH + fw;
                    host_filters[index] = (c + k) * (fw + fh);
                }
            }
        }
    }

    // Device memory allocations
    double *device_input_padded, *device_filters, *device_filters_flipped, *device_output;
    cudaMalloc(&device_input_padded, input_padded_bytes);
    cudaMalloc(&device_filters, filter_bytes);
    cudaMalloc(&device_filters_flipped, filter_bytes);
    cudaMalloc(&device_output, output_bytes);

    cudaMemcpy(device_input_padded, host_input_padded, input_padded_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filters, host_filters, filter_bytes, cudaMemcpyHostToDevice);

    // dimensions for blocks and grid
    dim3 block_size(16, 16);
    dim3 grid_size((INPUT_WIDTH + 15) / 16, (INPUT_HEIGHT + 15) / 16, NUM_FILTERS);

    //  ****** NAIVE CONVOLUTION STARTS HERE ******
    cudaEvent_t start_event_naive, stop_event_naive;
    cudaEventCreate(&start_event_naive);
    cudaEventCreate(&stop_event_naive);

    // naive conv timing
    cudaEventRecord(start_event_naive);
    conv_naive_kernel<<<grid_size, block_size>>>(device_input_padded, device_filters, device_output,
                                                 INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                                                 NUM_FILTERS, FILTER_HEIGHT, FILTER_WIDTH);
    cudaEventRecord(stop_event_naive);
    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_event_naive);

    float naive_kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&naive_kernel_time_ms, start_event_naive, stop_event_naive);

    // Compute checksum for naive convolution
    double output_checksum_naive = 0.0;
    for (int i = 0; i < NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH; ++i) {
        output_checksum_naive += host_output[i];
    }

    printf("C1_checksum: %.6f, C1_execution_time (s):  %.6f\n", output_checksum_naive, naive_kernel_time_ms / 1000);

    //  ****** TILED CONVOLUTION STARTS HERE ******

    cudaEvent_t start_event_tiled, stop_event_tiled;
    cudaEventCreate(&start_event_tiled);
    cudaEventCreate(&stop_event_tiled);

    // Measure time for tiled convolution
    cudaEventRecord(start_event_tiled);
    conv_tiled_kernel<<<grid_size, block_size>>>(device_input_padded, device_filters, device_output,
                                                 INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                                                 NUM_FILTERS, FILTER_HEIGHT, FILTER_WIDTH);
    cudaEventRecord(stop_event_tiled);
    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_event_tiled);

    float tiled_kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&tiled_kernel_time_ms, start_event_tiled, stop_event_tiled);

    // Compute checksum for tiled convolution
    double output_checksum_tiled = 0.0;
    for (int i = 0; i < NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH; ++i) {
        output_checksum_tiled += host_output[i];
    }

    printf("C2_checksum: %.6f, C2_execution_time (s):  %.6f\n", output_checksum_tiled, tiled_kernel_time_ms / 1000);

    //  ****** CUDNN CONVOLUTION STARTS HERE ******

    // cuDNN handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, NUM_FILTERS, INPUT_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, PADDING, PADDING, STRIDE, STRIDE, DILATION, DILATION, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));

    // cudnn can figure out the output dimensions of the convolution for us
    int N_out, C_out, H_out, W_out;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &N_out, &C_out, &H_out, &W_out));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N_out, C_out, H_out, W_out));


    // redefine dimensions since cudnn already takes care of padding
    size_t input_bytes = INPUT_CHANNELS * (INPUT_HEIGHT) * (INPUT_WIDTH) * sizeof(double);
    filter_bytes = NUM_FILTERS * INPUT_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH * sizeof(double);

    // keep in mind how the dimensions change after applying K filters
    output_bytes = NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double);

    // Host allocations

    // we remake the input without padding
    free(host_input_padded);
    double* host_input = (double*)malloc(input_bytes);

    // make the flipped filter allocation
    double* host_filters_flipped = (double*)malloc(filter_bytes);
    

    // Initialize input and filter (without padding this time) I[c, x, y] = c * (x + y)
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int x = 0; x < INPUT_HEIGHT; ++x) {
            for (int y = 0; y < INPUT_WIDTH; ++y) {
                host_input[c * INPUT_HEIGHT * INPUT_WIDTH + x * INPUT_WIDTH + y] = c * (x + y);
            }
        }
    }

    // flip the filters to poppulate host_filters_flipped
    for (int k = 0; k < NUM_FILTERS; ++k) {
        for (int c = 0; c < INPUT_CHANNELS; ++c) {
            for (int fh = 0; fh < FILTER_HEIGHT; ++fh) {
                for (int fw = 0; fw < FILTER_WIDTH; ++fw) {
                    // Calculate the flipped index for the filter
                    int flipped_index = ((k * INPUT_CHANNELS + c) * FILTER_HEIGHT + (FILTER_HEIGHT - 1 - fh)) * FILTER_WIDTH + (FILTER_WIDTH - 1 - fw);
                    int original_index = ((k * INPUT_CHANNELS + c) * FILTER_HEIGHT + fh) * FILTER_WIDTH + fw;
                    
                    // Copy the value from the original filter to the flipped filter
                    host_filters_flipped[flipped_index] = host_filters[original_index];
                }
            }
        }
    }

    // free up old memory and make new device memory allocations
    cudaFree(device_input_padded);
    cudaMalloc(&device_input_padded, input_bytes);
    cudaFree(device_filters);
    cudaMalloc(&device_filters, filter_bytes);
    cudaFree(device_filters_flipped);
    cudaMalloc(&device_filters_flipped, filter_bytes);
    cudaFree(device_output);
    cudaMalloc(&device_output, output_bytes);

    cudaMemcpy(device_input_padded, host_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filters, host_filters, filter_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filters_flipped, host_filters_flipped, filter_bytes, cudaMemcpyHostToDevice);


    // Select convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    int algo_count = 0;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        1, &algo_count, &perf_results));
    cudnnConvolutionFwdAlgo_t algo = perf_results.algo;

    // Allocate workspace
    size_t workspace_bytes = 0;
    void* d_workspace = nullptr;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        algo, &workspace_bytes));
    if (workspace_bytes > 0) {
        (cudaMalloc(&d_workspace, workspace_bytes));
    }

    // Perform convolution
    double alpha = 1.0, beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CHECK_CUDNN(cudnnConvolutionForward(
        handle, &alpha,
        input_desc, device_input_padded,
        filter_desc, device_filters_flipped,
        conv_desc, algo,
        d_workspace, workspace_bytes,
        &beta,
        output_desc, device_output));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cudnn_ms = 0.0f;
    cudaEventElapsedTime(&cudnn_ms, start, stop);
    
    // get result to host
    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);

    // cudnn checksum
    double cudnn_checksum = 0.0;
    for (int i = 0; i < output_bytes / sizeof(double); ++i) {
        cudnn_checksum += host_output[i];
    }

    printf("C3_checksum: %.6f, C3_execution_time (s):  %.6f\n", cudnn_checksum, cudnn_ms / 1000);

    // liberation
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(device_input_padded);
    cudaFree(device_filters);
    cudaFree(device_output);
    cudaFree(device_filters_flipped);
    free(host_input);
    free(host_filters);
    free(host_output);
    free(host_filters_flipped);

    return 0;
}
