.PHONY:all

TARGETS = cuda_kernels.o attention_kernels.o
# BIN = ../../bin

all:$(TARGETS)

#subdir1/makefile:
%.o : %.cu
	nvcc -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -c -o $@ $<

clean:
	rm *.o
