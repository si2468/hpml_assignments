#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_CHANNELS 3
#define INPUT_HEIGHT 1024
#define INPUT_WIDTH 1024
#define NUM_FILTERS 64
#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3
#define PADDING 1

#define TILE_WIDTH 16
#define SHARED_TILE_WIDTH (TILE_WIDTH + FILTER_WIDTH - 1)

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
    size_t input_padded_bytes = INPUT_CHANNELS * (INPUT_HEIGHT + 2 * PADDING) * (INPUT_WIDTH + 2 * PADDING) * sizeof(double);
    size_t filter_bytes = NUM_FILTERS * INPUT_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH * sizeof(double);
    size_t output_bytes = NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(double);

    double* host_input_padded = (double*)calloc(INPUT_CHANNELS * (INPUT_HEIGHT + 2 * PADDING) * (INPUT_WIDTH + 2 * PADDING), sizeof(double));
    double* host_filters = (double*)malloc(filter_bytes);
    double* host_output = (double*)malloc(output_bytes);

    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int y = 0; y < INPUT_HEIGHT; ++y) {
            for (int x = 0; x < INPUT_WIDTH; ++x) {
                int padded_index = (c * (INPUT_HEIGHT + 2 * PADDING) + (y + PADDING)) * (INPUT_WIDTH + 2 * PADDING) + (x + PADDING);
                host_input_padded[padded_index] = c * (x + y);
            }
        }
    }

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

    double *device_input_padded, *device_filters, *device_output;
    cudaMalloc(&device_input_padded, input_padded_bytes);
    cudaMalloc(&device_filters, filter_bytes);
    cudaMalloc(&device_output, output_bytes);

    cudaMemcpy(device_input_padded, host_input_padded, input_padded_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filters, host_filters, filter_bytes, cudaMemcpyHostToDevice);

    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_size((INPUT_WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
                   (INPUT_HEIGHT + TILE_WIDTH - 1) / TILE_WIDTH,
                   NUM_FILTERS);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    conv_tiled_kernel<<<grid_size, block_size>>>(device_input_padded, device_filters, device_output,
                                                 INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                                                 NUM_FILTERS, FILTER_HEIGHT, FILTER_WIDTH);
    cudaEventRecord(stop_event);

    cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_event);

    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);

    double output_checksum = 0.0;
    for (int i = 0; i < NUM_FILTERS * INPUT_HEIGHT * INPUT_WIDTH; ++i) {
        output_checksum += host_output[i];
    }

    printf("C2_checksum: %.3f, C2_execution_time (ms):  %.3f\n", output_checksum, kernel_time_ms);

    cudaFree(device_input_padded);
    cudaFree(device_filters);
    cudaFree(device_output);
    free(host_input_padded);
    free(host_filters);
    free(host_output);

    return 0;
}