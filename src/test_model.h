#pragma once
#include <cstdio>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <type_traits>
#include <sys/time.h>
#include "nccl.h"
#include "init.h"
#include "utils.h"
#include "layer/linear.h"
#include "layer/operator.h"
#include "layer/bert_layer.h"
#include "layer/bert.h"


template<typename T>
void nccl_test(const BertInitParam BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){
    T *d_A;
    cudaMalloc((void **)&d_A, 32 * 512 * 32 * 32 * sizeof(T));
    cudaMemset(d_A, 1, 32 * 512 * 32 * 32 * sizeof(T));

    ncclAllReduce(d_A, d_A, 32 * 512 * 32 * 32, SParams.nccldatatype, ncclSum, SParams.comm, DParams.stream);
    printf("Communication Warmup Test from Rank: %d\n", SParams.world_rank);
}

template<typename T>
void linear_test(const BertInitParam &BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){ // 
    
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num; // BertParams.max_seq_len * BertParams.max_batch_size;
    
    auto linear_row = Linear_Row<T>(SParams, k, n);
    auto linear_col = Linear_Col<T>(SParams, k, n); 

    T *input, *output;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * n);
    linear_row.forward(input, output, m);
    linear_col.forward(input, output, m);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i<100; i++){
        linear_row.FuncWrapVec[0].callback(DParams);
        linear_row.FuncWrapVec[1].callback(DParams);
        linear_col.FuncWrapVec[0].callback(DParams);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f, from Rank: %d. \n", milliseconds, SParams.world_rank);
}

template<typename T>
void layernorm_test(const BertInitParam &BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){ // 
    
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num; // BertParams.max_seq_len * BertParams.max_batch_size;
    
    auto layernorm = LayerNorm<T>(SParams, k);

    T *input, *output;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    
    layernorm.forward(input, output, m);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i<100; i++){
        layernorm.FuncWrapVec[0].callback(DParams);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f, from Rank: %d. \n", milliseconds, SParams.world_rank);
}

template<typename T>
void mlp_bert_test(const BertInitParam &BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num; 
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * k * 4 / SParams.world_size);
    auto mlp = Bert_MLP<T>(BertParams, SParams, k, n, workspace);
    vector<FuncWrap> vec = mlp.forward(input, output, m);
    printf("Vec Size: %d, \n", vec.size());
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i<100; i++){
        for(vector<FuncWrap>::iterator itr = vec.begin(); itr != vec.end(); itr++){
            (*itr).callback(DParams);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f, from Rank: %d. \n", milliseconds, SParams.world_rank);
}

template<typename T>
void mha_bert_test(const BertInitParam &BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num; 
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * n * 10 / SParams.world_size);

    auto mha = Bert_MHA<T>(BertParams, SParams, k, n, workspace);

    vector<FuncWrap> vec = mha.forward(input, output, BertParams.max_batch_size, BertParams.max_seq_len);
    printf("Vec Size: %d, \n", vec.size());
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i<100; i++){
        for(vector<FuncWrap>::iterator itr = vec.begin(); itr != vec.end(); itr++){
            (*itr).callback(DParams);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f, from Rank: %d. \n", milliseconds, SParams.world_rank);
}

template<typename T>
void test_encoder(const BertInitParam &BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num;
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * n * 20);

    auto encoder = Bert_Encoder<T>(BertParams, SParams, k, n, workspace);
    vector<FuncWrap> vec = encoder.forward(input, output, BertParams.max_batch_size, BertParams.max_seq_len);
    printf("Vec Size: %d, \n", vec.size());
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i<10; i++){
        for(vector<FuncWrap>::iterator itr = vec.begin(); itr != vec.end(); itr++){
            (*itr).callback(DParams);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f, from Rank: %d. \n", milliseconds, SParams.world_rank);
}

template<typename T>
void test_bert(const BertInitParam &BertParams, const StaticNvInitParam &SParams, const DynamicNvInitParam &DParams){
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num;
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * n * 20);

    auto model = Bert_Model<T>(BertParams, SParams, k, n, workspace);
    vector<FuncWrap> vec = model.forward(input, output, BertParams.max_batch_size, BertParams.max_seq_len);
    printf("Vec Size: %d, \n", vec.size());
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i<10; i++){
        for(vector<FuncWrap>::iterator itr = vec.begin(); itr != vec.end(); itr++){
            (*itr).callback(DParams);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f, from Rank: %d. \n", milliseconds, SParams.world_rank);
}