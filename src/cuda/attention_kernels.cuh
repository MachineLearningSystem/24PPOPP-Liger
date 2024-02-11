#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <assert.h>

template <typename T>
void add_QKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* Q,
  const T* bias_Q,
  T* K,
  const T* bias_K,
  T* V,
  const T* bias_V,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream);

template <typename T>
void add_fusedQKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* QKV,
  const T* qkv_bias,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream);
  
template <typename T>
void attn_softmax_kernelLauncher(
  T* buffer,
  const T* attr_mask,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const T scalar,
  cudaStream_t stream);

template <typename T>
void transpose_kernelLauncher(
  T* dst,
  T* src,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  cudaStream_t stream);
