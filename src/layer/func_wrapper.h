#pragma once
#include <cstdio>
#include <ostream>
#include <vector>
#include <iostream>
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "../context.h"

#include "../cuda/cuda_kernels.cuh"
#include "../cuda/attention_kernels.cuh"

// Basic operations

void GemmEx(DynamicNvInitParam DParams, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                const void *alpha, const void *A, cudaDataType Atype,  int lda, 
                const void *B, cudaDataType Btype, int ldb, const void *beta, 
                void *C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo)
{
    cublasGemmEx(DParams.handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                 B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}


void GemmStridedBatchedEx(DynamicNvInitParam DParams, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A, cudaDataType Atype, int lda, long long int strideA,
                            const void *B, cudaDataType Btype, int ldb, long long int strideB, const void *beta, 
                            void *C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo)
{
    cublasGemmStridedBatchedEx(DParams.handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA,
                            B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
}


void Allreduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, DynamicNvInitParam DParams)
{
    ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, DParams.stream);
}

void ReduceScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, DynamicNvInitParam DParams)
{
    ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, DParams.stream);
}

void Allgather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, DynamicNvInitParam DParams)
{
    ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, DParams.stream);
}

// attention kernels

template <typename T>
void Add_QKV_bias_transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V,
        const int batch_size, const int seq_len, const int head_num, const int size_per_head, DynamicNvInitParam DParams)
{
    add_QKV_bias_transpose_kernelLauncher<T>(q_buf, k_buf, v_buf, Q, bias_Q, K, bias_K, V, bias_V, batch_size, seq_len, head_num, size_per_head, DParams.stream);
}

template <typename T>
void Add_fusedQKV_bias_transpose_kernelLauncher(T* q_buf, T* k_buf, T* v_buf, T* QKV, const T* qkv_bias,
        const int batch_size, const int seq_len, const int head_num, const int size_per_head, DynamicNvInitParam DParams)
{
    add_fusedQKV_bias_transpose_kernelLauncher<T>(q_buf, k_buf, v_buf, QKV, qkv_bias, batch_size, seq_len, head_num, size_per_head, DParams.stream);
}

template <typename T>
void Attn_softmax_kernel(T* buffer, const T* attr_mask, const int batch_size, const int seq_len, const int head_num, const T scalar, DynamicNvInitParam DParams)
{
    attn_softmax_kernelLauncher<T>(buffer, attr_mask, batch_size, seq_len, head_num, scalar, DParams.stream);
}

template <typename T>
void Transpose_kernel(T* dst, T* src, const int batch_size, const int seq_len, const int head_num, const int size_per_head, DynamicNvInitParam DParams)
{
    transpose_kernelLauncher<T>(dst, src, batch_size, seq_len, head_num, size_per_head, DParams.stream);
}

// cuda kernels

template <typename T>
void Add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, DynamicNvInitParam DParams)
{
    add_bias_act_kernelLauncher<T>(out, bias, m, n, DParams.stream);
}

template <typename T>
void Add_bias_input_layernorm_kernelLauncher(T *out, const T *input_tensor, const T *bias, const T *gamma, const T *beta, int m, int n, DynamicNvInitParam DParams)
{
    add_bias_input_layernorm_kernelLauncher<T>(out, input_tensor, bias, gamma, beta, m, n, DParams.stream);
}

template <typename T>
void Add_bias_input_layernorm_2_kernelLauncher(const T *from_tensor, const T *gamma, const T *beta, const T *bias, T *output, T *norm_output_buf_,
                                               const int m, const int n, DynamicNvInitParam DParams)
{
    add_bias_input_layernorm_2_kernelLauncher<T>(from_tensor, gamma, beta, bias, output, norm_output_buf_, m, n, DParams.stream);
}

template <typename T>
void Add_bias_input_kernelLauncher(T *output, const T *bias, const T *input, const int m, const int n, DynamicNvInitParam DParams)
{
    add_bias_input_kernelLauncher<T>(output, bias, input, m, n, DParams.stream);
}

template <typename T>
void Layer_Norm(const T *from_tensor, const T *gamma, const T *beta, T *norm_from_tensor_buf_, const int m, const int n, DynamicNvInitParam DParams)
{
    layer_norm<T>(from_tensor, gamma, beta, norm_from_tensor_buf_, m, n, DParams.stream);
}