# Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference


### Install

```
make
```
Remember to change the architecture code in Makefile for your GPUs.

The prototype system depends on: MPI, CUDA, NCCL, cuBLAS nlohmann-json.

It is tested with these library versions:
MVAPICH2 version 2.3.7, 3.2.1, CUDA 11.3, 12.2, NCCL 2.15.5, 2.18.5-1.
nlohmann-json 3.11.2

### Usage

Liger requires a NVIDIA multi-GPU node and it is tested with V100 (NVLink Gen1) and A100 (PCIe) GPUs. In theory, Liger can act better in a node with weak interconnection.
```
NCCL_MAX_NCHANNELS=3 CUDA_DEVICE_MAX_CONNECTIONS=2 mpirun -np 2 ./main
```
To run this artifact in a new system, it is required to profile the kernel execution time at first.

1. Profile:
    test_profile()
2. Liger:
    test_schedule()
3. TP Baseline:
    test_no_schedule()

### Some Quick Results
The running results generated from the artifact in our V100（Nvlink Gen1）system using 2 GPUs. 

- Baseline_Result.txt
- Liger_Result.txt

### Some details of the Artifact

- You can configure the model, adjust the request rate, or the decomposition factor in main.cu. The default configuration comes from GLM-130B.
- [Section-3.2] The process of function assembly can be found at src/layer/bert.h, it adopts an iterative apporach to store kernel launch functions in a vector.
- [Section-3.3] The schedule algorithm is implemented in src/schedule.h/run_stream_sync. You can also find the hybrid synchronziation method [Section-3.4] and the contention factor [Section-3.5] here.
- We manage the kernel decompostion [Section-3.6] in src/init.h/Sample.


