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
class LayerNorm{
    public:
        int k;
        float eps = 1e-5; // always default
        T* gamma;
        T* beta;
        StaticNvInitParam SParams;
        vector<FuncWrap> FuncWrapVec;

        LayerNorm(const StaticNvInitParam &SParams, const int &k){
            this->k = k;
            this->SParams = SParams;
            cudaMalloc((void**)&this->gamma, k * sizeof(T));
            cudaMalloc((void**)&this->beta, k * sizeof(T));
        }

        vector<FuncWrap> forward(T* input, T* output, int m){
            this->FuncWrapVec.clear();
            using namespace std::placeholders;
            // (const T *from_tensor, const T *gamma, const T *beta, T *norm_from_tensor_buf_, const int m, const int n, NvInitParam func_params)
            auto callback_linear = std::bind(Layer_Norm<T>, input, this->gamma, this->beta, output, m, this->k, _1);
            this->FuncWrapVec.push_back(FuncWrap(0.1, COMPUTATION, callback_linear));
            return this->FuncWrapVec;
        }
};