#pragma once
#include <cstdio>
#include <cuda_runtime_api.h>
#include <iostream>
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nccl.h>
#include <vector>
#include <unistd.h>
#include "../init.h"
#include "func_wrapper.h"
#include "linear.h"
#include "operator.h"

using namespace std;

template <typename T>
class Bert_MLP{
    public:
        int k, n, intermediate_size;
        int mlp_ratio = 4;
        StaticNvInitParam SParam;
        T *linear_row_output;
        // , *activation_output;
        T *bias;        
        Linear_Col<T> *linear_col;
        Linear_Row<T> *linear_row;        
        vector<FuncWrap> FuncWrapVec;

        Bert_MLP(const BertInitParam &BertParams, const StaticNvInitParam &SParam, const int &k, const int &n, T *workspace)
        {
            /**
                General linear layer
            **/
            this->k = k;
            this->n = n;
            this->intermediate_size = n * this->mlp_ratio;
            this->linear_row_output = workspace;
            // this->activation_output = workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->intermediate_size;
            this->SParam = SParam;
            this->linear_col = new Linear_Col<T>(SParam, k, this->intermediate_size);
            this->linear_row = new Linear_Row<T>(SParam, this->intermediate_size, k);
            cudaMalloc((void**)&this->bias, this->intermediate_size * sizeof(T) / SParam.world_size);
        }

        vector<FuncWrap> forward(T* input, T* output, int m){
            this->FuncWrapVec.clear();
            using namespace std::placeholders;
            vector<FuncWrap> vec_tmp = this->linear_col->forward(input, this->linear_row_output, m);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            auto callback_act = std::bind(Add_bias_act_kernelLauncher<T>, this->linear_row_output, \
                                    this->bias, m, this->intermediate_size / this->SParam.world_size, _1);
            this->FuncWrapVec.push_back(FuncWrap(0.2, COMPUTATION, callback_act));

            vec_tmp = this->linear_row->forward(this->linear_row_output, output, m);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            return this->FuncWrapVec;
        }
};

template <typename T>
class Bert_MHA{
    public:

        int k, n;
        const T alpha = 1.0f;
        const T beta = 0.0f;
        T scalar;
        Linear_Col<T> *linear_q_col;
        Linear_Col<T> *linear_k_col;
        Linear_Col<T> *linear_v_col;
        Linear_Row<T> *linear_w_row;
        StaticNvInitParam SParam;
        BertInitParam BertParams;
        T* q_bias;
        T* k_bias;
        T* v_bias;
        T* w_bias;
        T *attr_mask;


        T *q_workspace, *k_workspace, *v_workspace, *q_buf_workspace, *k_buf_workspace, *v_buf_workspace;
        T *qk_buf_workspace, *attn_trans_out_workspace, *attn_out_workspace;

        vector<FuncWrap> FuncWrapVec;

        Bert_MHA(const BertInitParam &BertParams, const StaticNvInitParam &SParam, const int &k, const int &n, T *workspace){
            this->k = k;
            this->n = n;
            this->scalar = 1 / sqrtf(BertParams.size_per_head * 1.0f);
            this->SParam = SParam;
            this->BertParams = BertParams;

            this->linear_q_col = new Linear_Col<T>(SParam, this->k, this->n);
            this->linear_k_col = new Linear_Col<T>(SParam, this->k, this->n);
            this->linear_v_col = new Linear_Col<T>(SParam, this->k, this->n);
            this->linear_w_row = new Linear_Row<T>(SParam, this->k, this->n);

            this->q_workspace = workspace;
            this->k_workspace = this->q_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;
            this->v_workspace = this->k_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;
            
            this->q_buf_workspace = this->v_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;
            this->k_buf_workspace = this->q_buf_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;
            this->v_buf_workspace = this->k_buf_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;

            this->qk_buf_workspace = this->v_buf_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;
            this->attn_trans_out_workspace = this->qk_buf_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;
            this->attn_out_workspace = this->attn_trans_out_workspace + BertParams.max_batch_size * BertParams.max_seq_len * this->n / SParam.world_size;

            cudaMalloc((void**)&this->q_bias, sizeof(T) * this->k / SParam.world_size);
            cudaMalloc((void**)&this->k_bias, sizeof(T) * this->k / SParam.world_size);
            cudaMalloc((void**)&this->v_bias, sizeof(T) * this->k / SParam.world_size);
            cudaMalloc((void**)&this->attr_mask, sizeof(T) * BertParams.max_batch_size * BertParams.max_seq_len * BertParams.max_seq_len);
        }

        vector<FuncWrap> forward(T* input, T* output, int batch_size, int seq_len){
            this->FuncWrapVec.clear();
            using namespace std::placeholders;            
            // qkv linear

            // printf("qkv linear batch_size: %d, seq_len: %d, head_num: %d, size_per_head: %d\n", batch_size, seq_len, this->BertParams.head_num, this->BertParams.size_per_head);

            vector<FuncWrap> vec_tmp = this->linear_q_col->forward(input, this->q_workspace, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            vec_tmp = this->linear_k_col->forward(input, this->k_workspace, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            vec_tmp = this->linear_v_col->forward(input, this->v_workspace, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            // transpose
            // printf("add_qkv_bias batch_size: %d, seq_len: %d, head_num: %d, size_per_head: %d\n", batch_size, seq_len, this->BertParams.head_num, this->BertParams.size_per_head);
            auto add_qkv_bias = std::bind(Add_QKV_bias_transpose<T>, this->q_buf_workspace, this->k_buf_workspace, this->v_buf_workspace, this->q_workspace, \
                                    this->q_bias, this->k_workspace, this->k_bias, this->v_workspace, this->v_bias, batch_size, \
                                    seq_len, this->BertParams.head_num / this->SParam.world_size, this->BertParams.size_per_head, _1);
            this->FuncWrapVec.push_back(FuncWrap(0.2, COMPUTATION, add_qkv_bias));

            // qk
            auto qk_gemm = std::bind(GemmStridedBatchedEx, _1, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, this->BertParams.size_per_head, \
            &this->alpha, this->k_buf_workspace, this->SParam.datatype, this->BertParams.size_per_head, seq_len * this->BertParams.size_per_head, \
            this->q_buf_workspace, this->SParam.datatype, this->BertParams.size_per_head, seq_len * this->BertParams.size_per_head, \
            &this->beta, this->qk_buf_workspace, this->SParam.datatype, seq_len, seq_len * seq_len, batch_size * this->BertParams.head_num / this->SParam.world_size, \
            this->SParam.datatype, static_cast<cublasGemmAlgo_t>(-1));
            this->FuncWrapVec.push_back(FuncWrap(0.3, COMPUTATION, qk_gemm));

            // softmax
            auto softmax = std::bind(Attn_softmax_kernel<T>, this->qk_buf_workspace, this->attr_mask, batch_size, \
                        seq_len, this->BertParams.head_num / this->SParam.world_size, this->scalar, _1);
            this->FuncWrapVec.push_back(FuncWrap(0.2, COMPUTATION, softmax));

            // qkv
            auto qkv_gemm = std::bind(GemmStridedBatchedEx, _1, CUBLAS_OP_N, CUBLAS_OP_N, this->BertParams.size_per_head, seq_len, seq_len, \
                            &this->alpha, this->v_buf_workspace, this->SParam.datatype, this->BertParams.size_per_head, seq_len * this->BertParams.size_per_head, \
                            this->qk_buf_workspace, this->SParam.datatype, seq_len, seq_len * seq_len, &this->beta, this->attn_trans_out_workspace, \
                            this->SParam.datatype, this->BertParams.size_per_head, seq_len * this->BertParams.size_per_head, \
                            batch_size * this->BertParams.head_num / this->SParam.world_size, this->SParam.datatype, static_cast<cublasGemmAlgo_t>(-1));
            this->FuncWrapVec.push_back(FuncWrap(0.3, COMPUTATION, qkv_gemm));

            // transpose back
            auto transpose = std::bind(Transpose_kernel<T>, this->attn_out_workspace, this->attn_trans_out_workspace, batch_size, seq_len, \
                                    this->BertParams.head_num / this->SParam.world_size, this->BertParams.size_per_head, _1);
            this->FuncWrapVec.push_back(FuncWrap(0.1, COMPUTATION, transpose));

            // linear row
            vec_tmp = this->linear_w_row->forward(this->attn_out_workspace, output, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            return this->FuncWrapVec;
        }
};