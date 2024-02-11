#include "init.h"
#include "utils.h"
#include "layer/bert.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <map>
#include <queue>
#include <unistd.h>
#include <pthread.h>
#include <unordered_map>
#include <mutex>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>


template <class T, class MODELTYPE>
class Scheduler{
    private:
        MODELTYPE* _model;
        pthread_t _thread;
        vector<pair<int, int>> batch_seq_list;   
        // std::mutex batch_seq_list_mutex;  
    public:        
        bool tag = false;
        queue<Sample*> main_q;
        vector<Sample*> exe_vec;
        int schedule_deep = 6;
        StaticNvInitParam SParams;
        DynamicNvInitParam DParams0;
        DynamicNvInitParam DParams1;
        std::unordered_map<kernel_info, float, KeyHash> profiling_res_dict;
        std::unordered_map<kernel_info, std::vector<float>, KeyHash> profiling_decompse_res_dict;
        BertInitParam BertParams;
        vector<recorder> recs;

        T *input, *output;

        cudaEvent_t Main_EVENT, Wait_EVENT;


        Scheduler(Bert_Model<T> *model, const BertInitParam &BertParams, const StaticNvInitParam &SParams, T *input, T *output, \
                const std::string& main_filename, const std::string& sub_filename)
        {
            this->_model = model;
            this->SParams = SParams;
            this->BertParams = BertParams;

            this->input = input;
            this->output = output;
            
            DParams0 = DynamicNvInitParam(-4);
            DParams1 = DynamicNvInitParam(0);
            // DParams0 = DynamicNvInitParam();
            // DParams1 = DynamicNvInitParam();

            this->profiling_res_dict = MainFileToDict(main_filename);
            this->profiling_decompse_res_dict = SubFileToDict(sub_filename);

            cudaEventCreate(&Main_EVENT);
            cudaEventCreate(&Wait_EVENT);    
        }

        Scheduler(Bert_Model<T> *model, const BertInitParam &BertParams, const StaticNvInitParam &SParams, T *input, T *output)
        {
            this->_model = model;
            this->SParams = SParams;
            this->BertParams = BertParams;

            this->input = input;
            this->output = output;
            
            DParams0 = DynamicNvInitParam(-4);
            DParams1 = DynamicNvInitParam(0);
        }

        void profile(int low_bs, int high_bs, int low_sq, int high_sq, string &main_filename, string &sub_filename){
            cudaEvent_t start, stop;
            float elapsed;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            int itr = 20;

            for(int i = low_bs; i < high_bs; i++){
                for(int j = low_sq; j < high_sq; j++){
                    if((i * j) % 4 != 0){
                        continue;
                    }
                    if (this->SParams.world_rank == 0) printf("Batch Size: %d; Seq_Len: %d; \n", i, j);
                    vector<FuncWrap> vec = this->_model->forward(this->input, this->output, i, j);
                    for(int k = 0; k < this->BertParams.kernel_num ;k++){
                        kernel_info info = {k, i, j};
                        float time_min = 100000;
                        for(int test=0; test < 3; test++){
                            vec[k].callback(this->DParams0);
                            vec[k].callback(this->DParams0);
                            cudaDeviceSynchronize();
                            cudaEventRecord(start, this->DParams0.stream);
                            for(int m = 0; m < itr; m++){
                                vec[k].callback(this->DParams0);
                            }
                            cudaEventRecord(stop, this->DParams0.stream);
                            cudaEventSynchronize(stop);
                            cudaEventElapsedTime(&elapsed, start, stop);
                            if(elapsed < time_min){
                                time_min = elapsed;
                            }
                            usleep(10000);
                        }
                        this->profiling_res_dict[info] = time_min/itr;
                            
                        if(vec[k].type == GEMM || vec[k].type == COMMUNICATION){
                            vector<float> tmp_decompse;
                            for(int n = 1; n < BertParams.factor; n++){
                                FuncWrap fc;
                                if(vec[k].type == GEMM){
                                    fc = decompose_callback(vec[k], BertParams.factor, n);
                                } else {
                                    fc = decompose_comm_callback(vec[k], BertParams.factor, n);
                                }

                                float time_decompose_min = 100000;

                                for(int test=0; test < 3; test++){
                                
                                    fc.callback(this->DParams0);
                                    fc.callback(this->DParams0);
                                    cudaDeviceSynchronize();
                                    cudaEventRecord(start, this->DParams0.stream);
                                    for(int m = 0; m < itr; m++){
                                        fc.callback(this->DParams0);
                                    }
                                    cudaEventRecord(stop, this->DParams0.stream);
                                    cudaEventSynchronize(stop);
                                    cudaEventElapsedTime(&elapsed, start, stop);
                                    if(elapsed < time_min){
                                        time_decompose_min = elapsed;
                                    }
                                }
                                // if (this->SParams.world_rank == 0) printf("\t Name: %d ; Batch Size: %d; Seq_Len: %d; %d / %d; Time: %f. \n", k, i, j, n, BertParams.factor, elapsed);
                                tmp_decompse.push_back(time_decompose_min/itr);
                            }
                            this->profiling_decompse_res_dict[info] = tmp_decompse;
                            usleep(10000);
                        }
                    }
                }
            }
          
            if (this->SParams.world_rank == 0){
                MainDictToFile(this->profiling_res_dict, main_filename);
                SubDictToFile(this->profiling_decompse_res_dict, sub_filename);
            }
        }


        void serve(int batch_size, int seq_len){
            // use timestamp to record the input
            vector<FuncWrap> vec = this->_model->forward(this->input, this->output, batch_size, seq_len);
            
            for(int i = 0; i < this->BertParams.layer_num; i++){
                for(int j = 0; j < this->BertParams.kernel_num; j++){
                    vec[i * this->BertParams.kernel_num + j].name = j;
                    vec[i * this->BertParams.kernel_num + j].duration = this->profiling_res_dict[{j, batch_size, seq_len}];
                }
            }
            
            Sample *sample = new Sample(vec, this->profiling_decompse_res_dict, batch_size, seq_len, BertParams.factor);
            this->main_q.push(sample);
        }


        void update_vec(){
            if(!this->exe_vec.empty()){
                if(this->exe_vec[0]->empty()){
                    // if (this->SParams.world_rank == 0) {printf("Old Sample deleted. \n");}                                  
                    Sample* tmp = this->exe_vec[0];
                    this->recs.push_back(tmp->get_recording());
                    for(int i = 0; i < this->exe_vec.size() - 1; i++){
                        this->exe_vec[i] = this->exe_vec[i+1];
                    }
                    this->exe_vec[exe_vec.size() - 1] = tmp;
                    this->exe_vec.pop_back();
                }
            }
            
            // Add new one if waiting queue has samples and exe_vec has space.
            for(int i = this->exe_vec.size(); i < this->schedule_deep; i++){
                if(!this->main_q.empty()){
                    this->exe_vec.push_back(this->main_q.front());
                    this->main_q.pop();
                }else{
                    break;
                } 
            }

            if(this->exe_vec.empty()){
                usleep(100);
            }
        }

        void run_sequential(){
            this->tag = true;
            float time_planned = 0;
            cudaEventRecord(Wait_EVENT, DParams0.stream);
            while(this->tag){
                this->update_vec();
                if(!this->exe_vec.empty()){
                    cudaEventSynchronize(Wait_EVENT);   
                    while(!this->exe_vec.at(0)->empty()){
                        this->exe_vec.at(0)->dequeue(DParams0, 1, time_planned);
                    }     
                    cudaEventRecord(Wait_EVENT, DParams0.stream);                                  
                }
            }
            pthread_join(this->_thread, NULL); 
        }

        void run_stream_sync(){
            this->tag = true;
            float time_planned = 0.0;
            layer_type main_type;

            cudaEventRecord(Wait_EVENT, DParams0.stream);
            // cudaEventSynchronize(Wait_EVENT);
            while (this->tag) {
                this->update_vec();
                // print_time("test point 0"); 
                if(!this->exe_vec.empty()){
                    cudaEventSynchronize(Wait_EVENT);
                    cudaStreamWaitEvent(DParams1.stream, Main_EVENT, 0); 
                    while (!this->exe_vec.at(0)->empty()){
                        if (this->exe_vec.at(0)->is_switch()) {
                            main_type = this->exe_vec.at(0)->type();
                            cudaEventRecord(Wait_EVENT, DParams0.stream); 
                            time_planned += this->exe_vec.at(0)->dequeue(DParams0, 1, time_planned);  
                            // print_time("launch");                          
                            cudaEventRecord(Main_EVENT, DParams0.stream);                    
                            break;
                        }else{                            
                            main_type = this->exe_vec.at(0)->type();
                            time_planned += this->exe_vec.at(0)->dequeue(DParams0, 1, time_planned);                            
                        } 
                    }

                    // The contention factor     
                    time_planned *= 0.8;
                    // time_planned = 0;

                    // main stream has launched, traverse all sub samples for fullfilling.
                    for(size_t sub_size = 1; sub_size < this->exe_vec.size(); sub_size++){
                        while (time_planned > 0) {
                            if(!this->exe_vec.at(sub_size)->is_same_type(main_type)){                                
                                if(this->exe_vec.at(sub_size)->duration(0, time_planned) > time_planned){
                                    if(this->exe_vec.at(sub_size)->type() == GEMM)
                                    {   // only for gemm division, and it will not execution                                        
                                        if(!this->exe_vec.at(sub_size)->possible_duration(time_planned)){
                                            time_planned = 0.0;
                                        }
                                    }else{
                                        time_planned = 0.0;
                                    }
                                }else {
                                    time_planned -= this->exe_vec.at(sub_size)->dequeue(DParams1, 0, time_planned);
                                }                                
                            }
                            else {
                                break;
                            }
                        }                        
                    }
                    time_planned = 0.0;
                }             
            } 
            pthread_join(this->_thread, NULL); 
        }
       
        void run_stop(){
            this->tag = false;     
        }

        // Change this func to simulate the requests
        static void * simul_requests_helper(void * This){
            int times = 1;            
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            int rq_interval = ((Scheduler<T, MODELTYPE> *)This)->BertParams.rq_interval;
            sleep(1);
            MPI_Barrier(MPI_COMM_WORLD);
            ((Scheduler<T, MODELTYPE> *)This)->serve(2, 64);
            sleep(1);

            for(times = 0; times < 150; ){
                MPI_Barrier(MPI_COMM_WORLD);
                int bs = ((Scheduler<T, MODELTYPE> *)This)->BertParams.max_batch_size;
                int sq = (rand() % (((Scheduler<T, MODELTYPE> *)This)->BertParams.max_seq_len - 28)) + 16;
                if((bs * sq) % 4 == 0){
                    ((Scheduler<T, MODELTYPE> *)This)->serve(bs, sq);
                    times++;
                    usleep(rq_interval);
                }  
            }
            printf("Total %d requests. \n", times);
            sleep(5);

            ((Scheduler<T, MODELTYPE> *)This)->run_stop();
            return NULL;
        }

        void serving(){            
            int ret = 0;
            ret = pthread_create(&this->_thread, NULL, this->simul_requests_helper, this);

            if(ret == -1){
                printf("Create pthread error! \n");
            }else{
                printf("Create pthread success! \n");
            }            
        }
        

        // only select middle results to report
        void output_recording(){
            int times = 0;
            int ignore_num = 20;

            float total_avg_duration = 0.0;

            if (this->SParams.world_rank == 0){
                for (vector<recorder>::iterator itr = this->recs.begin() + ignore_num; itr != this->recs.end(); itr++){
                    float cuda_time = 0, cpu_time = 0;
                    cudaEventElapsedTime(&cuda_time, itr->start, itr->stop);
                    cpu_time = (itr->cpu_time - (this->recs.begin() + ignore_num)->cpu_time) * 1000;

                    printf("request: %d, CPU_Time: %f ms, CUDA_Time: %f ms. \n", times, cpu_time, cuda_time);

                    total_avg_duration += cpu_time + cuda_time;
                    times++;
                }

                float total_duration = 0;
                cudaEventElapsedTime(&total_duration, (this->recs.begin() + ignore_num)->start, (this->recs.end()-1)->stop);
                printf("Total_Duration: %f ms. \n", total_duration);
                printf("Throughput: %f samples/s. \n", 1000 * times / total_duration);
                printf("Average Latency: %f ms. \n", total_avg_duration / times);
                printf("Total %d records. \n", times);
            }                  
        }
};