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
EXECS :=  array_add_cpu array_add_gpu_non_unified array_add_gpu_unified
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

# this program contains a timer that we can easily use to track time during programs
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################

# this program contains CPU code to add two vectors and profile it
array_add_cpu: q1.cpp
	$(NVCC) $< -o $@ $(OPTIONS)

# this program contains a CUDA kernel to add two vectors with memory coelescing and profile it, using non-unified memory
array_add_gpu_non_unified: q2.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

# this program contains a CUDA kernel to add two vectors with memory coelescing and profile it, using unified memory
array_add_gpu_unified: q3.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################






