#include <string>
#include <mpi.h>
#include <iostream>
#include "src/communicator.h"
#include "src/init.h"
#include "src/test_model.h"
#include "src/test_schedule.h"
#include <cuda_fp16.h>

int main(int argc, char* argv[]){
    int world_size, world_rank;
    ncclComm_t ncclcomm = createNcclComm(argc, argv, world_size, world_rank);

    int max_batch_size = 2;
    int max_seq_len = 128;
    int head_num = 128;
    int size_per_head = 96;
    int layer_num = 4;
    int kernel_num = 16; // total number of kernels in a single layer
    int factor = 8; // decomposition factor
    int arr_rate = 80; // arrival_rate times/second
    

    // int max_batch_size = std::stoi(argv[1]);
    // int max_seq_len = std::stoi(argv[2]);
    // int head_num = std::stoi(argv[3]);
    // int size_per_head = std::stoi(argv[4]);
    // int layer_num = std::stoi(argv[5]);
    // int kernel_num = std::stoi(argv[6]);
    // int factor = std::stoi(argv[7]);
    // int arr_rate = std::stoi(argv[8]);

    int rq_interval = 1e6 / arr_rate; // us.

    StaticNvInitParam staticNvParam = StaticNvInitParam(world_rank, world_size, ncclcomm, 1);
    DynamicNvInitParam dynamicNvParam = DynamicNvInitParam();
    BertInitParam BertParams= BertInitParam(max_batch_size, max_seq_len, head_num, size_per_head, layer_num, kernel_num, factor, rq_interval);

    // some unit test
    // nccl_test<float>(BertParams, staticNvParam, dynamicNvParam);
    // linear_test<__half>(BertParams, staticNvParam, dynamicNvParam);
    // layernorm_test<__half>(BertParams, staticNvParam, dynamicNvParam);
    // mlp_bert_test<__half>(BertParams, staticNvParam, dynamicNvParam);
    // mha_bert_test<__half>(BertParams, staticNvParam, dynamicNvParam);
    // test_encoder<__half>(BertParams, staticNvParam, dynamicNvParam);
    // test_bert<__half>(BertParams, staticNvParam, dynamicNvParam);
    
    // to collect profiling results
    test_profile<__half>(BertParams, staticNvParam);
    // running with Liger
    test_schedule<__half>(BertParams, staticNvParam);
    // running without Liger
    test_no_schedule<__half>(BertParams, staticNvParam);
    // MPI_Finalize();
}