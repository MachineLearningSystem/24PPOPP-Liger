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
#include "context.h"
#include "utils.h"

struct recorder{
    double cpu_time;
    cudaEvent_t start;
    cudaEvent_t stop;
};

class Sample{
    private:
        std::vector<FuncWrap> _SubFuncWrapVec;
        int _factor = 0;
        int _remain = 0;
        int _sub_exe_num = 0;

        bool _sub_empty(){
            return this->_SubFuncWrapVec.size() == this->_sub_exe_num;
        }

        bool _sub_size(){
            return this->_SubFuncWrapVec.size() - this->_sub_exe_num;
        }

        void _sub_clear(){
            _SubFuncWrapVec.clear();
            this->_sub_exe_num = 0;
        }

    public:
        std::vector<FuncWrap> FuncWrapVec;
        std::unordered_map<kernel_info, std::vector<float>, KeyHash> profiling_decompse_res_dict;
        struct timeval sample_arrive, sample_start;
        size_t exe_num = 0;
        cudaEvent_t start, stop;
        int batch_size, seq_len;


        Sample(std::vector<FuncWrap>& FuncWrapVec, std::unordered_map<kernel_info, std::vector<float>, KeyHash> &profiling_decompse_res_dict, int batch_size, int seq_len, int factor){
            this->FuncWrapVec = FuncWrapVec;
            this->profiling_decompse_res_dict = profiling_decompse_res_dict;
            this->batch_size = batch_size;
            this->seq_len = seq_len;
            this->_factor = factor;
            // printf("%zu \n", this->FuncWrapVec->size());
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            this->record_arrival();
        }

        void print_info_debug(){
            std::vector<FuncWrap>::iterator itr;
            for(itr = FuncWrapVec.begin(); itr < FuncWrapVec.end(); itr++){
                printf("Name: %d; Time: %f. \n", itr->name, itr->duration);
                if(itr->type == GEMM){
                    std::vector<float> vec = this->profiling_decompse_res_dict[{itr->name, this->batch_size, this->seq_len}];
                    for(int i=0; i < vec.size(); i++){
                        printf("\t Time: %f. \n", vec[i]);
                    }
                }
                
            }
        }

        bool main_empty(){
            return this->FuncWrapVec.size() == this->exe_num;
        }

        bool empty(){
            return this->main_empty() && this->_sub_empty();
        }

        bool next_empty(){
            if(this->_sub_empty()){
                return this->FuncWrapVec.size() - 1 == this->exe_num;
            }else{
                if(this->main_empty()){
                    return this->_SubFuncWrapVec.size() - 1 == this->_sub_exe_num;
                }else {
                    return false;
                }
            }            
        }

        size_t size(){
            return this->FuncWrapVec.size() - this->exe_num + this->_SubFuncWrapVec.size() - this->_sub_exe_num;
        }

        bool is_switch_from_comp_to_comm(){
            if(this->_sub_empty()){
                return (COMMUNICATION == this->FuncWrapVec.at(this->exe_num + 1).type) && \
                    ((COMPUTATION == this->FuncWrapVec.at(this->exe_num).type) || (GEMM == this->FuncWrapVec.at(this->exe_num).type));
            }else if(this->_SubFuncWrapVec.size() - 1 == this->_sub_exe_num){
                return COMMUNICATION == this->FuncWrapVec.at(this->exe_num).type;
            }else{
                return false;
            }
        }

        bool is_switch_from_comm_to_comp(){
            if(this->_sub_empty()){
                return (COMMUNICATION == this->FuncWrapVec.at(this->exe_num).type) && \
                    ((COMPUTATION == this->FuncWrapVec.at(this->exe_num + 1).type) || (GEMM == this->FuncWrapVec.at(this->exe_num + 1).type));
            }else{
                return false;
            }
        }

        bool is_switch(){
            bool b_tmp = false;
            if(this->next_empty()){
                b_tmp = true;
            }else {
                b_tmp = this->is_switch_from_comp_to_comm() || this->is_switch_from_comm_to_comp();
            }
            return b_tmp;
        }

        bool is_same_type(layer_type type){
            bool tmp = false;
            if(this->_sub_empty()){
                if(type == COMMUNICATION && this->FuncWrapVec.at(this->exe_num).type == COMMUNICATION){
                    tmp = true;
                }

                if(type != COMMUNICATION && this->FuncWrapVec.at(this->exe_num).type != COMMUNICATION){
                    tmp = true;
                }
            }else {
                if(type == COMMUNICATION){
                    tmp = false;
                }else{
                    tmp = true;
                }                
            }
            return tmp;
        }

        layer_type type(){
            if(this->_sub_empty()){
                return this->FuncWrapVec.at(this->exe_num).type;
            }else{
                return GEMM;
            }
        }

        // ToDo: Fit the Sub_queue
        float dequeue(DynamicNvInitParam & DParams, int is_main, float &time_planned){
            float duration = 0.0;

            if (this->exe_num == 0) {
                this->record_start();
                this->record_gpu_start(DParams);
            }

            if (this->next_empty()) {
                this->record_gpu_stop(DParams);
            }

            if(!this->_sub_empty()){
                if(this->_sub_exe_num == 0) {
                    this->exe_num++;
                }
                this->_SubFuncWrapVec.at(this->_sub_exe_num).callback(DParams);
                duration = this->_SubFuncWrapVec.at(this->_sub_exe_num).duration;
                this->_sub_exe_num++;
            }else{
                if(!this->empty())
                {   
                    this->FuncWrapVec.at(this->exe_num).callback(DParams);                
                    duration = this->FuncWrapVec.at(this->exe_num).duration;

                    if(is_main == 0 && this->FuncWrapVec.at(this->exe_num).type == COMMUNICATION)
                    {
                        duration = time_planned;
                    }
                    this->exe_num++; 
                }
            }

            return duration;
        }

        recorder get_recording(){
            recorder rec = {this->time_cpu_elapsed(), this->start, this->stop};
            return rec;
        }

        float duration(int is_main, float &time_planned){  
            float duration = 0.0;

            if(is_main == 0 && this->FuncWrapVec.at(this->exe_num).type == COMMUNICATION)
            {
                return time_planned;
            }


            if(this->_sub_empty()){
                return this->FuncWrapVec.at(this->exe_num).duration;
            }else {
                duration = this->_SubFuncWrapVec.at(this->_sub_exe_num).duration;                
                return duration;
            }                        
        }

        bool is_second_comm(){
            if(this->_sub_empty()){
                return (COMMUNICATION == this->FuncWrapVec.at((this->exe_num + 1)% this->FuncWrapVec.size()).type) ;
            }else{
                return false;
            }
        }
      
        bool possible_duration(float &time_planned){
            // return false;
            std::vector<float> vec;
            if(this->_sub_empty()){
                // 0 - 2; 0 - 6
                vec = this->profiling_decompse_res_dict[{this->FuncWrapVec.at(this->exe_num).name, this->batch_size, this->seq_len}];
                for(int i = this->_factor - 2; i >= 0; i--){
                    if(time_planned >= vec[i]){
                        this->_sub_clear();
                        int n_run = i + 1;
                        this->_remain = this->_factor - n_run;
                        this->_SubFuncWrapVec.push_back(decompose_callback(this->FuncWrapVec.at(this->exe_num), this->_factor, n_run, vec[i]));
                        this->_SubFuncWrapVec.push_back(decompose_callback(this->FuncWrapVec.at(this->exe_num), this->_factor, this->_remain, vec[this->_remain-1]));
                        // exe_num++;
                        return true;
                    }
                }
                return false;           
            }else{
                vec = this->profiling_decompse_res_dict[{this->_SubFuncWrapVec.at(this->_sub_exe_num).name, this->batch_size, this->seq_len}];                
                for(int i = this->_remain - 2; i >= 0; i--){
                    if(time_planned >= vec[i]){ // i != 0
                        int n_run = i + 1;
                        this->_remain -= n_run;
                        FuncWrap fw = this->_SubFuncWrapVec[1];
                        this->_sub_clear();
                        this->_SubFuncWrapVec.push_back(decompose_callback(fw, this->_factor, n_run, vec[1]));
                        this->_SubFuncWrapVec.push_back(decompose_callback(fw, this->_factor, this->_remain, vec[this->_remain-1]));
                        return true;
                    }
                }
                return false;
            }
        }

        void record_arrival(){
            gettimeofday(&this->sample_arrive, 0);
        }

        void record_start(){
            gettimeofday(&this->sample_start, 0);
        }

        void record_gpu_start(DynamicNvInitParam & DParams){            
            cudaEventRecord(start, DParams.stream);    
        }

        void record_gpu_stop(DynamicNvInitParam & DParams){
            cudaEventRecord(stop, DParams.stream);
        }

        double time_cpu_elapsed(){
            long seconds = this->sample_start.tv_sec - this->sample_arrive.tv_sec;
            long microseconds = this->sample_start.tv_usec - this->sample_arrive.tv_usec;
            double elapsed = seconds + microseconds*1e-6;
            return elapsed;
        }
};







// template<typename T>
// struct DenseWeight{
//     T* kernel;
//     T* bias;
// };

// template<typename T>
// struct LayerNormWeight{
//     T* gamma;
//     T* beta;
// };

// template<typename T>
// struct AttentionWeight{
//     DenseWeight<T> query_weight;
//     DenseWeight<T> key_weight;
//     DenseWeight<T> value_weight;
//     DenseWeight<T> attention_output_weight;
//     T* attr_mask;
// };

// template<typename T>
// struct FFNWeight{
//     DenseWeight<T> intermediate_weight;
//     DenseWeight<T> output_weight;
// };

// template <typename T>
// LayerNormWeight<T> init_layernorm_Weight(BertInitParam &BertParams, NvInitParam &NvParams){
//     LayerNormWeight<T> self_layernorm;
//     int head_num = BertParams.head_num;
//     int size_per_head = BertParams.size_per_head;
//     int world_size = NvParams.world_size;

//     T *gamma, *beta;    
//     cudaMalloc((void**)&gamma, head_num * size_per_head * sizeof(T));
//     cudaMalloc((void**)&beta, head_num * size_per_head * sizeof(T));
    
//     self_layernorm.gamma = gamma;
//     self_layernorm.beta = beta;

//     return self_layernorm;
// }

// template <typename T>
// AttentionWeight<T> init_attention_Weight(BertInitParam &BertParams, NvInitParam &NvParams){
//     AttentionWeight<T> self_attention;
//     int max_batch_size = BertParams.max_batch_size;
//     int max_seq_len = BertParams.max_seq_len;
//     int head_num = BertParams.head_num;
//     int size_per_head = BertParams.size_per_head;    
//     int world_size = NvParams.world_size;

//     int k = head_num * size_per_head;
//     int n = k;
//     T *q_kernel, *key_kernel, *value_kernel, *q_bias, *key_bias, *value_bias;

//     cudaMalloc((void**)&q_kernel, k * k * sizeof(T) / world_size);
//     cudaMalloc((void**)&key_kernel, k * k * sizeof(T) / world_size);
//     cudaMalloc((void**)&value_kernel, k * k * sizeof(T) / world_size);
//     cudaMalloc((void**)&q_bias, n * sizeof(T) / world_size);
//     cudaMalloc((void**)&key_bias, n * sizeof(T) / world_size);
//     cudaMalloc((void**)&value_bias, n * sizeof(T) / world_size);

//     self_attention.query_weight.kernel = q_kernel;
//     self_attention.key_weight.kernel = key_kernel;
//     self_attention.value_weight.kernel = value_kernel;
//     self_attention.query_weight.bias = q_bias;
//     self_attention.key_weight.bias = key_bias;
//     self_attention.value_weight.bias = value_bias;

//     T *linear_kernel, *linear_bias, *attr_mask;
//     cudaMalloc((void**)&linear_kernel, k * n * sizeof(T) / world_size);
//     cudaMalloc((void**)&linear_bias, n * sizeof(T) / world_size);
//     cudaMalloc((void**)&attr_mask,  max_batch_size * max_seq_len * sizeof(T) / world_size);

//     self_attention.attention_output_weight.kernel = linear_kernel;
//     self_attention.attention_output_weight.bias = linear_bias;
//     self_attention.attr_mask = attr_mask;

//     return self_attention;
// }

// template <typename T>
// FFNWeight<T> init_ffn_Weight(BertInitParam &BertParams, NvInitParam &NvParams){
//     FFNWeight<T> ffn;
//     int max_batch_size = BertParams.max_batch_size;
//     int max_seq_len = BertParams.max_seq_len;
//     int head_num = BertParams.head_num;
//     int size_per_head = BertParams.size_per_head;    
//     int world_size = NvParams.world_size;

//     int k = head_num * size_per_head;
//     int n = k;
//     int n1 = k * 4;

//     T *linear0_weight, *linear0_bias, *linear1_weight, *linear1_bias;
//     cudaMalloc((void**)&linear0_weight, k * n1 * sizeof(T) / world_size);
//     cudaMalloc((void**)&linear0_bias, n1 * sizeof(T) / world_size);
//     cudaMalloc((void**)&linear1_weight, n1 * n  * sizeof(T) / world_size);
//     cudaMalloc((void**)&linear1_bias, n * sizeof(T) / world_size);

//     ffn.intermediate_weight.kernel = linear0_weight;
//     ffn.intermediate_weight.bias = linear0_bias;
//     ffn.output_weight.kernel = linear1_weight;
//     ffn.output_weight.bias = linear1_bias;

//     return ffn;
// }