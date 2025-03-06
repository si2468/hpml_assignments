import numpy as np
import sys
import time

# for a simple loop
def dp(N,A,B):
    R = 0.0
    for j in range(0,N):
        R += A[j]*B[j]
    return R

def main():
    N = int(sys.argv[1])
    repetitions = int(sys.argv[2])

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    total_time = 0.0
    second_half_total_time = 0.0
    second_half_start = repetitions // 2

    for i in range(repetitions):
        start = time.monotonic()
        result = dp(N, A, B)
        end = time.monotonic()
        total_time += (end - start)


        if i >= second_half_start:
            second_half_total_time += (end - start)


    second_half_mean_time = second_half_total_time / second_half_start
    bandwidth = (2.0 * N * 4) / (second_half_mean_time * 1e9) 
    flops = (2.0 * N) / second_half_mean_time 

    print(f"N: {N} <T>: {second_half_mean_time:.6f} sec B: {bandwidth:.3f} GB/sec F: {flops:.3f} FLOP/sec")

if __name__ == "__main__":
    main()

