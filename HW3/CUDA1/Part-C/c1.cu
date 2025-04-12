#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Tensor dimensions
#define INPUT_CHANNELS 3
#define INPUT_HEIGHT 1024
#define INPUT_WIDTH 1024
#define NUM_FILTERS 64
#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3
#define PADDING 1

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

int main() {
    size_t input_padded_bytes = INPUT_CHANNELS * (INPUT_HEIGHT + 2 * PADDING) * (INPUT_WIDTH + 2 * PADDING) * sizeof(double);
    size_t filter_bytes = NUM_FILTERS * INPUT_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH * sizeof(double);
    size_t output_bytes = NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double);

    // Host allocations
    double* host_input_padded = (double*)calloc(INPUT_CHANNELS * (INPUT_HEIGHT + 2 * PADDING) * (INPUT_WIDTH + 2 * PADDING), sizeof(double));
    double* host_filters = (double*)malloc(filter_bytes);
    double* host_output = (double*)malloc(output_bytes);

    // Generate input: I[c, x, y] = c * (x + y)
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int y = 0; y < INPUT_HEIGHT; ++y) {
            for (int x = 0; x < INPUT_WIDTH; ++x) {
                int padded_index = (c * (INPUT_HEIGHT + 2 * PADDING) + (y + PADDING)) * (INPUT_WIDTH + 2 * PADDING) + (x + PADDING);
                host_input_padded[padded_index] = c * (x + y);
            }
        }
    }

    // Generate filters: F[k, c, i, j] = (c + k) * (i + j)
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
    double *device_input_padded, *device_filters, *device_output;
    cudaMalloc(&device_input_padded, input_padded_bytes);
    cudaMalloc(&device_filters, filter_bytes);
    cudaMalloc(&device_output, output_bytes);

    cudaMemcpy(device_input_padded, host_input_padded, input_padded_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filters, host_filters, filter_bytes, cudaMemcpyHostToDevice);

    // Kernel config
    dim3 block_size(16, 16);
    dim3 grid_size((INPUT_WIDTH + 15) / 16, (INPUT_HEIGHT + 15) / 16, NUM_FILTERS);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    conv_naive_kernel<<<grid_size, block_size>>>(device_input_padded, device_filters, device_output,
                                                 INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                                                 NUM_FILTERS, FILTER_HEIGHT, FILTER_WIDTH);
    cudaEventRecord(stop_event);

    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_event);

    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

    // Compute checksum
    double output_checksum = 0.0;
    for (int i = 0; i < NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH; ++i) {
        output_checksum += host_output[i];
    }

    printf("C1_checksum: %.3f, C1_execution_time (ms): %.3f\n", output_checksum, kernel_time_ms);

    // Cleanup
    cudaFree(device_input_padded);
    cudaFree(device_filters);
    cudaFree(device_output);
    free(host_input_padded);
    free(host_filters);
    free(host_output);

    return 0;
}
