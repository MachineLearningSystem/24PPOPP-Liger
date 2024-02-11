#pragma once
#include <cstdio>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include "utils.h"


inline ncclComm_t createNcclComm(int argc, char* argv[], int& world_size, int& world_rank){

    // MPI_Init(&argc, &argv);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    // Get MPI process information
    // int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < world_size) {
        printf("Error: This program requires at least %d GPUs\n", world_size);
        MPI_Finalize();
    }    
    cudaSetDevice(world_rank);
    // NCCL Initialization

    ncclUniqueId id;
    ncclComm_t comm;
    if (world_rank == 0) {
        ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    NCCLCHECK(ncclCommInitRank(&comm, world_size, id, world_rank));
    // MPI_Finalize();
    printf("Communication Complete from Rank: %d\n", world_rank);
    return comm;
}