#pragma once
#include <cstdio>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/select.h>
#include <type_traits>
#include <sys/time.h>
#include "nccl.h"
#include <functional>
#include <vector>
#include <unordered_map>

class DynamicNvInitParam{
    public:
        // for cuda
        cudaStream_t stream;
        cublasHandle_t handle;

        DynamicNvInitParam(){      
            cudaStreamCreate(&this->stream); 
            cublasCreate(&this->handle);
            cublasSetStream(handle, stream);
        }

        DynamicNvInitParam(int prioirty){      
            cudaStreamCreateWithPriority(&this->stream, 1, prioirty); 
            cublasCreate(&this->handle);
            cublasSetStream(handle, stream);
        }
}; 

class StaticNvInitParam{
    public:
        // int algo;
        // for nccl
        int world_size;
        int world_rank;
        ncclDataType_t nccldatatype;
        cudaDataType_t datatype;
        ncclComm_t comm;
        StaticNvInitParam(){};
        StaticNvInitParam(const int &world_rank, const int &world_size, const ncclComm_t &comm, int datatype){
     
            this->world_rank = world_rank;
            this->world_size = world_size;
            this->comm = comm;

            if(datatype == 0){
                this->datatype = CUDA_R_32F;
                this->nccldatatype = ncclFloat;
            }else if(datatype == 1){
                this->datatype = CUDA_R_16F;
                this->nccldatatype = ncclHalf;
            }
        }
}; 

class BertInitParam{
    /* weights for masked_multi_head_attention */
    // LayerNormWeight<T> self_layernorm;
    // AttentionWeight<T> self_attention;
    // LayerNormWeight<T> ffn_layernorm;
    // FFNWeight<T> ffn;
    public:
        int max_batch_size;
        int max_seq_len;
        int head_num;
        int size_per_head;
        int layer_num;
        int kernel_num;
        int factor;
        int rq_interval;
        BertInitParam(){};
        BertInitParam(int &max_batch_size, int &max_seq_len, int &head_num, int &size_per_head, int &layer_num, int &kernel_num, int &factor, int &rq_interval){
            this->max_batch_size = max_batch_size;
            this->max_seq_len = max_seq_len;
            this->head_num = head_num;
            this->size_per_head = size_per_head;
            this->layer_num = layer_num;
            this->kernel_num = kernel_num;
            this->factor = factor;
            this->rq_interval = rq_interval;
        }
};

enum layer_type {COMMUNICATION, COMPUTATION, GEMM};

typedef std::tuple<cublasOperation_t, cublasOperation_t, int, int, int,
                const void*, const void*, cudaDataType,  int, 
                const void*, cudaDataType, int, const void*, 
                void*, cudaDataType, int, cudaDataType, cublasGemmAlgo_t> GEMM_ARGS;

typedef std::tuple<const void*, void*, int, ncclDataType_t, ncclRedOp_t, ncclComm_t> COMM_ARGS;

class FuncWrap{
    // int order;
    public:
        float duration;
        layer_type type;
        std::function<void(DynamicNvInitParam)> callback;
        GEMM_ARGS args_gemm_tuple;
        COMM_ARGS args_comm_tuple;
        FuncWrap(){};
        int name;
        
        FuncWrap(int name, float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback){
            this->name = name;
            this->duration = duration;
            this->type = type;
            this->callback = callback;
        }

        FuncWrap(int name, float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback, GEMM_ARGS args_gemm_tuple){
            this->name = name;
            this->duration = duration;
            this->type = type;
            this->callback = callback;
            this->args_gemm_tuple = args_gemm_tuple;
        }

        FuncWrap(float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback){
            this->duration = duration;
            this->type = type;
            this->callback = callback;
        }
        
        FuncWrap(float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback, GEMM_ARGS args_gemm_tuple){
            this->duration = duration;
            this->type = type;
            this->callback = callback;
            this->args_gemm_tuple = args_gemm_tuple;
        }

        FuncWrap(float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback, COMM_ARGS args_comm_tuple){
            this->duration = duration;
            this->type = type;
            this->callback = callback;
            this->args_comm_tuple = args_comm_tuple;
        }

        FuncWrap(int name, float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback, COMM_ARGS args_comm_tuple){
            this->name = name;
            this->duration = duration;
            this->type = type;
            this->callback = callback;
            this->args_comm_tuple = args_comm_tuple;
        }

        bool not_comm(){
            return !(type == COMMUNICATION);
        }
};

struct kernel_info {
    int kernel_name;
    int batch_size;
    int seq_len;

    bool operator==(const kernel_info& other) const {
        return std::tie(kernel_name, batch_size, seq_len) == std::tie(other.kernel_name, other.batch_size, other.seq_len);
    }
};

struct KeyHash {
    std::size_t operator()(const kernel_info& key) const {
        return std::hash<int>{}(key.kernel_name) ^ std::hash<int>{}(key.batch_size) ^ std::hash<double>{}(key.seq_len);
    }
};