#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mkl_cblas.h>

float bdp(long N, float *pA, float *pB) {
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R;
}

int main(int argc, char *argv[]) {

    long N = atol(argv[1]);
    int repetitions = atoi(argv[2]);

    float *pA = (float *)malloc(N * sizeof(float));
    float *pB = (float *)malloc(N * sizeof(float));
    if (!pA || !pB) {
        fprintf(stderr, "malloc failed.\n");
        return -1;
    }

    for (long i = 0; i < N; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    struct timespec start, end;
    double total_time = 0.0;
    double second_half_total_time = 0.0;
    int second_half_start = repetitions / 2;

    for (int i = 0; i < repetitions; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        volatile float result = bdp(N, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &end);
        result += 1.0;

        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += elapsed;

        if (i >= second_half_start){
            second_half_total_time += elapsed;
        }

    }

    //printf("total time is %f %f\n", total_time, second_half_total_time);

    //double mean_time = total_time / repetitions;

    // mean time for second half of runs
    double second_half_mean_time = second_half_total_time / (repetitions - second_half_start);

    // bandwidth for second half of runs - 2 floating point each with 4 bytes
    double bandwidth = (2.0 * N * sizeof(float)) / (second_half_mean_time* 1e9);

    // throughput for second half of runs
    double flops = (2.0 * N) / second_half_mean_time; 

    printf("N: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n", N, second_half_mean_time, bandwidth, flops);

    free(pA);
    free(pB);
    return EXIT_SUCCESS;
}
