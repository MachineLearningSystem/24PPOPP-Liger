#pragma once
#include <cstdio>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <string>
#include <type_traits>
#include <sys/time.h>
#include <unistd.h>
#include "nccl.h"
#include "init.h"
#include "utils.h"
#include "layer/bert.h"
#include "schedule.h"



template<typename T>
void test_schedule(const BertInitParam &BertParams, const StaticNvInitParam &SParams){
    
    cudaSetDevice(SParams.world_rank);
    // Init Model
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num;
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * n * 80);

    // ToDo: is a single workspace feasible

    Bert_Model<T> * model = new Bert_Model<T>(BertParams, SParams, k, n, workspace);
    string main_filename = "./Main_" + to_string(SParams.world_size) + "_" + to_string(BertParams.size_per_head) \
                        + '_' + to_string(BertParams.head_num) + "_" + to_string(BertParams.max_batch_size) + '_' + to_string(BertParams.max_seq_len) + ".json";
    string sub_filename = "./Sub_" + to_string(SParams.world_size) + "_" + to_string(BertParams.size_per_head) \
                        + '_' + to_string(BertParams.head_num) + "_" + to_string(BertParams.max_batch_size) + '_' + to_string(BertParams.max_seq_len) + ".json";
    auto scheduler = Scheduler<T, Bert_Model<T>>(model, BertParams, SParams, input, output, main_filename, sub_filename);

    
    scheduler.serving();
    scheduler.run_stream_sync(); 
    scheduler.run_stop();
    scheduler.output_recording();
}


template<typename T>
void test_no_schedule(const BertInitParam &BertParams, const StaticNvInitParam &SParams){
    
    cudaSetDevice(SParams.world_rank);
    // Init Model
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num;
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * n * 80);

    // ToDo: is a single workspace feasible

    Bert_Model<T> * model = new Bert_Model<T>(BertParams, SParams, k, n, workspace);
    string main_filename = "./Main_" + to_string(SParams.world_size) + "_" + to_string(BertParams.size_per_head) \
                        + '_' + to_string(BertParams.head_num) + "_" + to_string(BertParams.max_batch_size) + '_' + to_string(BertParams.max_seq_len) + ".json";
    string sub_filename = "./Sub_" + to_string(SParams.world_size) + "_" + to_string(BertParams.size_per_head) \
                        + '_' + to_string(BertParams.head_num) + "_" + to_string(BertParams.max_batch_size) + '_' + to_string(BertParams.max_seq_len) + ".json";
    auto scheduler = Scheduler<T, Bert_Model<T>>(model, BertParams, SParams, input, output, main_filename, sub_filename);

    
    scheduler.serving();    
    scheduler.run_sequential(); 
    scheduler.run_stop();
    scheduler.output_recording();
}

template<typename T>
void test_profile(const BertInitParam &BertParams, const StaticNvInitParam &SParams){
    
    cudaSetDevice(SParams.world_rank);
    // Init Model
    int k = BertParams.size_per_head * BertParams.head_num;
    int m = BertParams.max_seq_len * BertParams.max_batch_size;
    int n = BertParams.size_per_head * BertParams.head_num;
    
    T *input, *output, *workspace;
    cudaMalloc((void**)&input, sizeof(T) * m * k);
    cudaMalloc((void**)&output, sizeof(T) * m * k);
    cudaMalloc((void**)&workspace, sizeof(T) * m * n * 100);

    // ToDo: is a single workspace feasible
    Bert_Model<T> * model = new Bert_Model<T>(BertParams, SParams, k, n, workspace);    
    auto scheduler = Scheduler<T, Bert_Model<T>>(model, BertParams, SParams, input, output);
    
    string main_filename = "./Main_" + to_string(SParams.world_size) + "_" + to_string(BertParams.size_per_head) \
                        + '_' + to_string(BertParams.head_num) + "_" + to_string(BertParams.max_batch_size) + '_' + to_string(BertParams.max_seq_len) + ".json";
    string sub_filename = "./Sub_" + to_string(SParams.world_size) + "_" + to_string(BertParams.size_per_head) \
                        + '_' + to_string(BertParams.head_num) + "_" + to_string(BertParams.max_batch_size) + '_' + to_string(BertParams.max_seq_len) + ".json";
    // scheduler.profile(1, 9, 1, 257, main_filename, sub_filename);
    scheduler.profile(1, BertParams.max_batch_size + 1, 1, BertParams.max_seq_len + 1, main_filename, sub_filename);
}