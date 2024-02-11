.PHONY:main

main : main.cu
	nvcc -lmpi -lnccl -lcublas \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	src/cuda/cuda_kernels.cu src/cuda/attention_kernels.cu main.cu \
	-o main

# main : main.cpp
# 	nvcc -lmpi -lnccl -lcublas -gencode=arch=compute_70,code=sm_70 \
# 	-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 \
# 	-I src/cuda cuda_kernels.o main.o -o main

# main : main.o cuda_kernels.o attention_kernels.o
# 	nvcc -lmpi -lnccl -lcublas -gencode=arch=compute_70,code=sm_70 \
# 	-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 \
# 	main.o main.o cuda_kernels.o attention_kernels.o -o main

# TARGETS = cuda_kernels.o attention_kernels.o

# $(TARGETS):
# 	make -C src/cuda
# 	@cp -f src/cuda/cuda_kernels.o /home/nsccgz_jiangsu/projects/Liger
# 	@cp -f src/cuda/attention_kernels.o /home/nsccgz_jiangsu/projects/Liger


# main.o : main.cpp
# 	nvcc -lmpi -lnccl -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75  -c main.cpp

clean:
	rm main