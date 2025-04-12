###
### For COMS E6998 Spring 2024
### Instructor: Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS := conv_naive conv_tiled conv_cudnn
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

## Naive Convolution Kernel - compute the convolution using a naive CUDA kernel
conv_naive : c1.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS) -lcudnn

## Tiled Convolution Kernel - compute the convolution using shared memory tiling
conv_tiled : c2.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS) -lcudnn

## CuDNN Convolution Kernel - compute the convolution using the cuDNN library
conv_cudnn : c3.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS) -lcudnn

#######################################################################






