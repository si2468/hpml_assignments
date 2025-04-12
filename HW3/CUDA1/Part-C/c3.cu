#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

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

int main() {
    //  ****** CUDNN CONVOLUTION ******

    size_t filter_bytes = NUM_FILTERS * INPUT_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH * sizeof(double);
    size_t output_bytes = NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double);

    // Host allocations with padding
    double* host_filters = (double*)malloc(filter_bytes);
    double* host_output = (double*)malloc(output_bytes);

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
    double *device_input_padded, *device_filters_flipped, *device_output;

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

    // Host allocations

    // we make the input without padding
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
    cudaMalloc(&device_input_padded, input_bytes);
    cudaMalloc(&device_filters_flipped, filter_bytes);
    cudaMalloc(&device_output, output_bytes);

    cudaMemcpy(device_input_padded, host_input, input_bytes, cudaMemcpyHostToDevice);
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

    printf("C3_checksum: %.3f, C3_execution_time (ms):  %.3f\n", cudnn_checksum, cudnn_ms);

    // liberation
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(device_input_padded);
    cudaFree(device_output);
    cudaFree(device_filters_flipped);
    free(host_input);
    free(host_filters);
    free(host_output);
    free(host_filters_flipped);

    return 0;
}
