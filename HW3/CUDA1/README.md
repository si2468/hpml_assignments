# CUDA1 Assignment - Spring 2024

## **Overview**

This repository contains the code for a CUDA-based assignment for the course COMS E6998 at Columbia University, Spring 2024. The assignment involves various CUDA programming tasks, including vector addition, matrix multiplication, and convolution, leveraging both CPU and GPU computation.

The `Makefile` provided automates the compilation and execution of multiple CUDA programs, each demonstrating different GPU programming techniques. The code uses both basic and expanded kernels to implement operations like vector addition and matrix multiplication, and also includes optimizations such as memory management.

## **Prerequisites**

Before running the CUDA programs, ensure the following software is installed:

- **CUDA Toolkit (v10.0 or higher recommended)**: You can download it from [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).
- **NVIDIA Driver**: Ensure that the appropriate driver for your GPU is installed. This is required to run CUDA programs.
- **Make**: Ensure you have `make` installed on your system.
- **Linux-based OS**: The Makefile is configured for Linux. Ensure that you are running a Linux-based OS, such as Ubuntu or Debian.
  
### **Optional but Recommended**

- **CUDNN**: For the convolution kernel, it's recommended to have `cudnn` installed. This library provides high-performance primitives for deep learning tasks.

## **Directory Structure**

The project consists of the following files:

- `Makefile`: The build system that compiles all CUDA programs.
- `timer.cu`, `timer.h`: Provides a timer function to measure the execution time of CUDA programs.
- `vecadd.cu`, `vecaddKernel.h`: Implements basic vector addition.
- `vecaddKernel00.cu`, `vecaddKernel01.cu`: Different versions of the vector addition kernel, used for comparison.
- `matmult.cu`, `matmultKernel.h`: Implements basic matrix multiplication.
- `matmultKernel00.cu`, `matmultKernel01.cu`: Different matrix multiplication kernels, with optimizations.
- `array_add.cpp`: Performs array addition using a CPU implementation.
- `array_add_gpu_non_unified.cu`, `array_add_gpu_unified.cu`: GPU-based array addition implementations.
- `convolution.cu`: Implements a naive convolution kernel.

## **How to Use the Makefile**

The provided `Makefile` is designed to automate the process of compiling and running the CUDA programs. It defines various targets that correspond to different steps in the process, such as compilation, cleaning, and creating archives.

### **Available Targets**

Here are the primary targets defined in the `Makefile` and their purpose:

1. **`make` (default target)**:
   - Running `make` will compile all the CUDA programs listed under the `EXECS` variable. This includes compiling both basic and optimized versions of the kernels and generating the corresponding executable files.

   **Example**:
   ```bash
   make
## **Cleaning the Project**

After compiling the project, you may want to clean up the generated object files and executables. The `make clean` target in the `Makefile` allows you to do this easily, and this will delete all *.o files and executables. 

### **Usage**

To compile all the executables in the project, run `make`. The executables can then we ran as shown below. 

After compiling the project, you may want to clean up the generated object files and executables. The `make clean` target in the `Makefile` allows you to do this easily, and this will delete all *.o files and executables. 

## **Available Executables**

### Group 1: Vector Addition Experiments (Memory Coalescing)

1. **`vecadd00`** - Vector Addition (Basic Kernel)  
   Performs vector addition using a basic CUDA kernel.  
   ➤ To be timed and compared with **`vecadd01`**.

2. **`vecadd01`** - Vector Addition (Optimized Kernel)  
   Performs vector addition using an optimized CUDA kernel with memory coalescing.  
   ➤ To be timed and compared with **`vecadd00`**.

---

### Group 2: Matrix Multiplication Experiments (Thread coarsening and tiling)

3. **`matmult01`** - Matrix Multiplication (Tiled with Thread Coarsening)  
   Performs matrix multiplication using an optimized kernel with tiling, shared memory, and thread coarsening.  
   ➤ Each thread computes **4 output elements**.  
   ➤ To be timed and compared with **`matmult00`**.

4. **`matmult00`** - Matrix Multiplication (Tiled)  
   Performs matrix multiplication using a basic CUDA kernel with tiling and shared memory.  
   ➤ Each thread computes **1 output element**.  
   ➤ To be timed and compared with **`matmult01`**.

---

### Group 3: Array Addition Experiments (CPU, GPU, GPU w/ Unified Memory)

5. **`array_add`** - Array Addition (CPU Implementation)  
   Performs array addition on the CPU.  
   ➤ Accepts array size in millions as an argument.  
   ➤ Can be compared with the GPU versions for performance.

6. **`array_add_gpu_non_unified`** - Array Addition (GPU, Non-Uniform Memory Access)  
   Performs array addition on the GPU with standard memory management.  
   ➤ Accepts array size in millions as an argument.  
   ➤ To be compared with **`array_add`** and **`array_add_gpu_unified`**.

7. **`array_add_gpu_unified`** - Array Addition (GPU, Unified Memory)  
   Performs array addition on the GPU using **unified memory**.  
   ➤ Accepts array size in millions as an argument.  
   ➤ To be compared with **`array_add`** and **`array_add_gpu_non_unified`**.

---

### Group 4: Convolution Experiments (Naive, Tiling, cuDNN)

8. **`conv`** - Convolution  
   Runs **three** versions of convolution:  
   - Naive CUDA kernel  
   - Optimized CUDA kernel with shared memory and tiling  
   - cuDNN's built-in convolution  
   ➤ Use this to compare the performance of naive, optimized, and cuDNN convolution implementations.


### Bash Commands to Run the Executables

```
# Group 1: Vector Addition
./vecadd00
./vecadd01

# Group 2: Matrix Multiplication
./matmult00
./matmult01

# Group 3: Array Addition
./array_add 10                # Run with an array of 10 million elements
./array_add_gpu_non_unified 10  # Run with an array of 10 million elements
./array_add_gpu_unified 10      # Run with an array of 10 million elements

# Group 4: Convolution
./conv


