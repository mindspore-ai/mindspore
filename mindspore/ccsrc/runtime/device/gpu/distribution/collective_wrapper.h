/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mpi.h>
#include <nccl.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "runtime/device/gpu/distribution/mpi_wrapper.h"
#include "runtime/device/gpu/distribution/nccl_wrapper.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

using MPIWrapper = mindspore::device::gpu::MPIWrapper;
using NCCLWrapper = mindspore::device::gpu::NCCLWrapper;

extern "C" EXPORT_WRAPPER void InitMPI();
extern "C" EXPORT_WRAPPER int local_rank_id();
extern "C" EXPORT_WRAPPER void InitNCCLComm();
extern "C" EXPORT_WRAPPER bool CreateCommGroup(const std::string &group_name, const std::vector<unsigned int> &ranks);
extern "C" EXPORT_WRAPPER int GetRankIDByGroup(const std::string &group_name);
extern "C" EXPORT_WRAPPER int GetGroupSize(const std::string &group_name);
extern "C" EXPORT_WRAPPER bool DestroyGroup(const std::string &group_name);
extern "C" EXPORT_WRAPPER ncclResult_t AllReduce(const void *input_addr, void *output_addr, size_t count,
                                                 ncclDataType_t data_type, ncclRedOp_t reduce_type, cudaStream_t stream,
                                                 const std::string &group);
extern "C" EXPORT_WRAPPER ncclResult_t AllGather(const void *input_addr, void *output_addr, size_t count,
                                                 ncclDataType_t data_type, cudaStream_t stream,
                                                 const std::string &group);
extern "C" EXPORT_WRAPPER ncclResult_t ReduceScatter(const void *input_addr, void *output_addr, size_t count,
                                                     ncclDataType_t data_type, ncclRedOp_t reduce_type,
                                                     cudaStream_t stream, const std::string &group);
extern "C" EXPORT_WRAPPER ncclResult_t Broadcast(const void *input_addr, void *output_addr, size_t count,
                                                 ncclDataType_t data_type, int root, cudaStream_t stream,
                                                 const std::string &group);
extern "C" EXPORT_WRAPPER ncclResult_t Send(const void *send_addr, size_t count, ncclDataType_t data_type,
                                            int peer_rank, cudaStream_t stream, const std::string &group_name);
extern "C" EXPORT_WRAPPER ncclResult_t Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank,
                                            cudaStream_t stream, const std::string &group_name);
extern "C" EXPORT_WRAPPER ncclResult_t GroupStart();
extern "C" EXPORT_WRAPPER ncclResult_t GroupEnd();
extern "C" EXPORT_WRAPPER std::vector<int> GetGroupRanks(const std::string &group_name);
