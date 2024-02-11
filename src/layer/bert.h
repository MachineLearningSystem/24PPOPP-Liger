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
#include "bert_layer.h"
#include "operator.h"

using namespace std;

template <typename T>
class Bert_Encoder{
    int k, n;
    StaticNvInitParam SParam;
    BertInitParam BertParam;

    Bert_MHA<T> *MHA;
    Bert_MLP<T> *MLP;
    LayerNorm<T> *ln_mha;
    LayerNorm<T> *ln_mlp;

    T *attn_workspace;
    T *mlp_workspace;
    T *mha_out_workspace;
    T *mlp_out_workspace;
    T *ln_mha_out_workspace;

    vector<FuncWrap> FuncWrapVec;

    public:
        Bert_Encoder(const BertInitParam &BertParams, const StaticNvInitParam &SParam, const int &k, const int &n, T *workspace){
            this->BertParam = BertParams;
            this->SParam = SParam;
            this->k = k;
            this->n = n;

            // inner workspace
            this->attn_workspace = workspace;
            this->mlp_workspace = this->attn_workspace + sizeof(T) * BertParams.max_batch_size * BertParams.max_seq_len * n * 10 / SParam.world_size;
            // for this class
            this->mha_out_workspace = this->mlp_workspace + sizeof(T) * BertParams.max_batch_size * BertParams.max_seq_len * n * 4 / SParam.world_size;
            this->ln_mha_out_workspace = this->mha_out_workspace + sizeof(T) * BertParams.max_batch_size * BertParams.max_seq_len * n;
            this->mlp_out_workspace = this->ln_mha_out_workspace + sizeof(T) * BertParams.max_batch_size * BertParams.max_seq_len * n;
            
            
            this->MHA = new Bert_MHA<T>(BertParams, SParam, k, n, this->attn_workspace);
            this->MLP = new Bert_MLP<T>(BertParams, SParam, k, n, this->mlp_workspace);
            this->ln_mha = new LayerNorm<T>(SParam, k);
            this->ln_mlp = new LayerNorm<T>(SParam, k);
        }

        vector<FuncWrap> forward(T* input, T* output, int batch_size, int seq_len){
            this->FuncWrapVec.clear();
            using namespace std::placeholders;

            // printf("MHA batch_size: %d, seq_len: %d\n", batch_size, seq_len);

            vector<FuncWrap> vec_tmp = this->MHA->forward(input, this->mha_out_workspace, batch_size, seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            // ToDo: Add bias layer norm
            vec_tmp = this->ln_mha->forward(this->mha_out_workspace, this->ln_mha_out_workspace, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());
            
            vec_tmp = this->MLP->forward(this->ln_mha_out_workspace, this->mlp_out_workspace, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            vec_tmp = this->ln_mlp->forward(this->mlp_out_workspace, output, batch_size * seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            return this->FuncWrapVec;
        }
};


template <typename T>
class Bert_Model{
    public:
        int k, n;
        StaticNvInitParam SParam;
        BertInitParam BertParams;
        vector<Bert_Encoder<T>*> encoders;
        vector<FuncWrap> FuncWrapVec;

        Bert_Model(const BertInitParam &BertParams, const StaticNvInitParam &SParam, const int &k, const int &n, T *workspace){
            this->k = k;
            this->n = n;
            this->SParam = SParam;
            this->BertParams = BertParams;

            for(int i = 0; i < BertParams.layer_num; i++){
                Bert_Encoder<T> *encoder = new Bert_Encoder<T>(BertParams, SParam, k, n, workspace);
                encoders.push_back(encoder);
            }
        }

        vector<FuncWrap> forward(T* input, T* output, int batch_size, int seq_len){
            this->FuncWrapVec.clear();
            // printf("Bert_Model batch_size: %d, seq_len: %d\n", batch_size, seq_len);
            vector<FuncWrap> vec_tmp = this->encoders[0]->forward(input, output, batch_size, seq_len);
            this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());

            for(int i = 1; i < this->BertParams.layer_num; i++){
                vector<FuncWrap>  vec_tmp = this->encoders[i]->forward(output, output, batch_size, seq_len);
                this->FuncWrapVec.insert(this->FuncWrapVec.end(), vec_tmp.begin(), vec_tmp.end());
            }
            return FuncWrapVec;
        }


};