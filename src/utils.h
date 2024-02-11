#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <cublas_v2.h>
#include <tuple>
#include "context.h"
#include "layer/func_wrapper.h"


#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

// #ifndef SAFE_QUEUE
// #define SAFE_QUEUE

// #include <queue>
// #include <mutex>
// #include <condition_variable>


// // A threadsafe-queue.
// template <class T>
// class SafeQueue
// {
// public:
//   SafeQueue(void)
//     : q()
//     , m()
//     , c()
//   {}

//   ~SafeQueue(void)
//   {}

//   // Add an element to the queue.
//   void enqueue(T t)
//   {
//     std::lock_guard<std::mutex> lock(m);
//     q.push(t);
//     c.notify_one();
//   }

//   // Get the "front"-element.
//   // If the queue is empty, wait till a element is avaiable.
//   T dequeue(void)
//   {
//     std::unique_lock<std::mutex> lock(m);
//     while(q.empty())
//     {
//       // release lock as long as the wait and reaquire it afterwards.
//       c.wait(lock);
//     }
//     T val = q.front();
//     q.pop();
//     return val;
//   }

// private:
//   std::queue<T> q;
//   mutable std::mutex m;
//   std::condition_variable c;
// };
// #endif

#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>



void MainDictToFile(const std::unordered_map<kernel_info, float, KeyHash>& data, const std::string& filename) {
    std::stringstream ss;
    ss << "[";
    for (const auto& entry : data) {
        ss << "{";
        ss << "\"kernel_name\":" << entry.first.kernel_name << ",";
        ss << "\"batch_size\":" << entry.first.batch_size << ",";
        ss << "\"seq_len\":" << entry.first.seq_len << ",";
        ss << "\"duration\":" << entry.second;
        ss << "},";
    }
    ss.seekp(-1, std::ios_base::end);
    ss << "]";
        
    std::ofstream file(filename);
    if (file.is_open()) {
        file << ss.str();
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

std::unordered_map<kernel_info, float, KeyHash> MainFileToDict(const std::string& filename) {
    std::ifstream file(filename);
    std::string json_str;

    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        json_str = buffer.str();
        file.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }

    std::unordered_map<kernel_info, float, KeyHash> dictionary;
    auto json = nlohmann::json::parse(json_str);
    for (const auto& item : json.items()) {
        auto value = item.value();  // 获取迭代器中的值

        kernel_info keyInfo;
        keyInfo.kernel_name = value.at("kernel_name").get<int>();
        keyInfo.batch_size = value.at("batch_size").get<int>();
        keyInfo.seq_len = value.at("seq_len").get<int>();

        float time = value.at("duration").get<float>();

        dictionary[keyInfo] = time;
    }

    return dictionary;
}

void SubDictToFile(const std::unordered_map<kernel_info, std::vector<float>, KeyHash> &data, const std::string& filename) {
    std::stringstream ss;
    ss << "[";
    for (const auto& entry : data) {
      ss << "{";
      ss << "\"kernel_name\":" << entry.first.kernel_name << ",";
      ss << "\"batch_size\":" << entry.first.batch_size << ",";
      ss << "\"seq_len\":" << entry.first.seq_len << ",";
      ss << "\"duration\": [";
      for (const auto& value : entry.second) {
        ss << value << ",";
      }
      // Remove the trailing comma
      ss.seekp(-1, std::ios_base::end);
      ss << "]";
      ss << "},";
    }
  // Remove the trailing comma
    ss.seekp(-1, std::ios_base::end);
    ss << "]";
    
    std::ofstream file(filename);
    if (file.is_open()) {
      file << ss.str();
      file.close();
    } else {
      std::cerr << "Unable to open file " << filename << std::endl;
    }
}

std::unordered_map<kernel_info, std::vector<float>, KeyHash> SubFileToDict(const std::string& filename) {
  std::ifstream file(filename);
  std::string json_str;
  if(file.is_open()) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    json_str = buffer.str();
    file.close();
  } else {
    std::cerr << "Unable to open file " << filename << std::endl;
  }
  
  std::unordered_map<kernel_info, std::vector<float>, KeyHash> dictionary;
  auto json = nlohmann::json::parse(json_str);
  for (const auto& item : json.items())
  {
    auto value = item.value();
    kernel_info keyInfo;
    keyInfo.kernel_name = value.at("kernel_name").get<int>();
    keyInfo.batch_size = value.at("batch_size").get<int>();
    keyInfo.seq_len = value.at("seq_len").get<int>();
    std::vector<float> times = value.at("duration").get<std::vector<float>>();
    dictionary[keyInfo] = times;
  }
  return dictionary;
}


FuncWrap decompose_callback(FuncWrap &old_func, int &factor, int &n, float duration = 0.0){
  using namespace std::placeholders;  
  auto callback = std::bind(GemmEx, _1,
                         std::get<0>(old_func.args_gemm_tuple),
                         std::get<1>(old_func.args_gemm_tuple),
                         std::get<2>(old_func.args_gemm_tuple) * n / factor,
                         std::get<3>(old_func.args_gemm_tuple),
                         std::get<4>(old_func.args_gemm_tuple),
                         std::get<5>(old_func.args_gemm_tuple),
                         std::get<6>(old_func.args_gemm_tuple),
                         std::get<7>(old_func.args_gemm_tuple),
                         std::get<8>(old_func.args_gemm_tuple),
                         std::get<9>(old_func.args_gemm_tuple),
                         std::get<10>(old_func.args_gemm_tuple),
                         std::get<11>(old_func.args_gemm_tuple),
                         std::get<12>(old_func.args_gemm_tuple),
                         std::get<13>(old_func.args_gemm_tuple),
                         std::get<14>(old_func.args_gemm_tuple),
                         std::get<15>(old_func.args_gemm_tuple),
                         std::get<16>(old_func.args_gemm_tuple),
                         std::get<17>(old_func.args_gemm_tuple));
  
  auto args_gemm = std::make_tuple( std::get<0>(old_func.args_gemm_tuple),
                         std::get<1>(old_func.args_gemm_tuple),
                         std::get<2>(old_func.args_gemm_tuple) * n / factor,
                         std::get<3>(old_func.args_gemm_tuple),
                         std::get<4>(old_func.args_gemm_tuple),
                         std::get<5>(old_func.args_gemm_tuple),
                         std::get<6>(old_func.args_gemm_tuple),
                         std::get<7>(old_func.args_gemm_tuple),
                         std::get<8>(old_func.args_gemm_tuple),
                         std::get<9>(old_func.args_gemm_tuple),
                         std::get<10>(old_func.args_gemm_tuple),
                         std::get<11>(old_func.args_gemm_tuple),
                         std::get<12>(old_func.args_gemm_tuple),
                         std::get<13>(old_func.args_gemm_tuple),
                         std::get<14>(old_func.args_gemm_tuple),
                         std::get<15>(old_func.args_gemm_tuple),
                         std::get<16>(old_func.args_gemm_tuple),
                         std::get<17>(old_func.args_gemm_tuple));
  
  return FuncWrap(old_func.name, duration, GEMM, callback, args_gemm);
}

FuncWrap decompose_comm_callback(FuncWrap &old_func, int &factor, int &n, float duration = 0.0){
  using namespace std::placeholders;  
  std::vector<FuncWrap>  new_funcs;
  auto callback = std::bind(Allreduce, 
                            std::get<0>(old_func.args_comm_tuple),
                            std::get<1>(old_func.args_comm_tuple),
                            std::get<2>(old_func.args_comm_tuple) * n / factor,
                            std::get<3>(old_func.args_comm_tuple),
                            std::get<4>(old_func.args_comm_tuple),
                            std::get<5>(old_func.args_comm_tuple), _1);
  
  auto args_comm = std::make_tuple(std::get<0>(old_func.args_comm_tuple),
                            std::get<1>(old_func.args_comm_tuple),
                            std::get<2>(old_func.args_comm_tuple) * n / factor,
                            std::get<3>(old_func.args_comm_tuple),
                            std::get<4>(old_func.args_comm_tuple),
                            std::get<5>(old_func.args_comm_tuple));

  return FuncWrap(old_func.name, duration, COMMUNICATION, callback, args_comm);
}

void warmup_half(ncclComm_t comm, cudaStream_t stream_nccl){
    __half *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, 32 * 512 * 32 * 32 * sizeof(__half));
    cudaMalloc((void **)&d_B, 32 * 512 * 32 * 32 * sizeof(__half));
    cudaMalloc((void **)&d_C, 32 * 512 * 32 * 32 * sizeof(__half));
    cudaMemset(d_A, 1, 32 * 512 * 32 * 32 * sizeof(__half));
    cudaMemset(d_B, 1, 32 * 512 * 32 * 32 * sizeof(__half));
    cudaMemset(d_C, 1, 32 * 512 * 32 * 32 * sizeof(__half));
    ncclAllReduce(d_A, d_A, 32 * 512 * 32 * 32, ncclHalf, ncclSum, comm, stream_nccl);
    ncclAllReduce(d_B, d_B, 32 * 512 * 32 * 32, ncclHalf, ncclSum, comm, stream_nccl);
    ncclAllReduce(d_C, d_C, 32 * 512 * 32 * 32, ncclHalf, ncclSum, comm, stream_nccl);
}

void print_time(char *str){
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // MPI_Barrier(MPI_COMM_WORLD);
  auto now = std::chrono::system_clock::now();

  // 转换为时间戳
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);

  // 转换为tm结构
  auto now_tm = *std::localtime(&now_c);

  // 获取自午夜以来的毫秒数
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) % 1000;

  // 打印格式化的时间
  std::cout << "world rank" << rank << "  "  << str << ":  " << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
  std::cout << '.' << std::setfill('0') << std::setw(3) << milliseconds.count() << std::endl;
}


// float duration, layer_type type, std::function<void(DynamicNvInitParam)> callback, GEMM_ARGS args_gemm_tuple