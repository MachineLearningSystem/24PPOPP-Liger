#pragma once
#include <cstdio>
#include <cuda_runtime_api.h>
#include <iostream>
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <unistd.h>
#include "../init.h"
#include "func_wrapper.h"

using namespace std;

template <typename T>
class Linear_Row{
    /**
    need to All_Reduce in Linear_Row
    **/
    private:    
        const int funcNum = 2;
    public:
        // the number of input and output features matrix size is (m*k)(k*n)
        int k, n;
        T *weights;
        StaticNvInitParam SParams;
        const T alpha = 1.0f;
        const T beta = 0.0f;
        vector<FuncWrap> FuncWrapVec;

        Linear_Row(const StaticNvInitParam &SParams, const int &k, const int &n)
        {
            /**
                General linear layer
            **/
            this->k = k;
            this->n = n;
            this->SParams = SParams;
            cudaMalloc((void**)&this->weights, k * n * sizeof(T) / SParams.world_size); 
        }

        vector<FuncWrap> forward(T* input, T* output, int m){
            this->FuncWrapVec.clear();
            using namespace std::placeholders;
            
            auto callback_linear = std::bind(GemmEx, _1, CUBLAS_OP_N, CUBLAS_OP_N, this->n, m, this->k / this->SParams.world_size, &this->alpha, 
                                this->weights, this->SParams.datatype, this->n, input, this->SParams.datatype, this->k / this->SParams.world_size , &this->beta, 
                                output, this->SParams.datatype, this->n, this->SParams.datatype, static_cast<cublasGemmAlgo_t>(-1));
            auto args_gemm = std::make_tuple(CUBLAS_OP_N, CUBLAS_OP_N, this->n, m, this->k / this->SParams.world_size, &this->alpha, 
                                this->weights, this->SParams.datatype, this->n, input, this->SParams.datatype, this->k / this->SParams.world_size , &this->beta, 
                                output, this->SParams.datatype, this->n, this->SParams.datatype, static_cast<cublasGemmAlgo_t>(-1));
            this->FuncWrapVec.push_back(FuncWrap(2.0, GEMM, callback_linear, args_gemm));

            if(this->SParams.world_size != 1){
                auto callback_comm = std::bind(Allreduce, output, output, m * this->n, this->SParams.nccldatatype, ncclSum, this->SParams.comm, _1);
                auto args_comm = std::make_tuple(output, output, m * this->n, this->SParams.nccldatatype, ncclSum, this->SParams.comm);
                
                this->FuncWrapVec.push_back(FuncWrap(2.0, COMMUNICATION, callback_comm, args_comm));
            }
            
            return this->FuncWrapVec;
        }
};



template <typename T>
class Linear_Col{
    /**
    need to All_Reduce in Linear_Row
    **/
    private:    
        const int funcNum = 1;
    public:
        // the number of input and output features matrix size is (m*k)(k*n)
        int k, n;
        T *weights;
        StaticNvInitParam SParams;
        const T alpha = 1.0f;
        const T beta = 0.0f;
        vector<FuncWrap> FuncWrapVec;

        Linear_Col(const StaticNvInitParam &SParams, const int &k, const int &n)
        {
            /**
                General linear layer
            **/
            this->k = k;
            this->n = n;
            this->SParams = SParams;
            cudaMalloc((void**)&this->weights, k * n * sizeof(T) / SParams.world_size); 
        }

        vector<FuncWrap> forward(T* input, T* output, int m){
            this->FuncWrapVec.clear();
            using namespace std::placeholders;
            
            auto callback_linear = std::bind(GemmEx, _1, CUBLAS_OP_N, CUBLAS_OP_N, this->n / this->SParams.world_size, m, this->k, &this->alpha,
                                    this->weights, this->SParams.datatype, this->n / this->SParams.world_size, input, this->SParams.datatype, this->k, &this->beta, 
                                    output, this->SParams.datatype, this->n / this->SParams.world_size, this->SParams.datatype, static_cast<cublasGemmAlgo_t>(-1));
            auto args_gemm = std::make_tuple(CUBLAS_OP_N, CUBLAS_OP_N, this->n / this->SParams.world_size, m, this->k, &this->alpha,
                                    this->weights, this->SParams.datatype, this->n / this->SParams.world_size, input, this->SParams.datatype, this->k, &this->beta, 
                                    output, this->SParams.datatype, this->n / this->SParams.world_size, this->SParams.datatype, static_cast<cublasGemmAlgo_t>(-1));
            this->FuncWrapVec.push_back(FuncWrap(1.0, GEMM, callback_linear, args_gemm));

            return this->FuncWrapVec;
        }
};