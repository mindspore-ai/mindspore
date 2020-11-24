/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include "runtime/device/gpu/distribution/collective_wrapper.h"

void InitMPI() { MPIWrapper::instance(); }

int local_rank_id() { return MPIWrapper::instance().local_rank_id(); }

void InitNCCLComm() { NCCLWrapper::instance().InitNCCLComm(); }

bool CreateCommGroup(const std::string &group_name, const std::vector<unsigned int> &ranks) {
  return MPIWrapper::instance().CreateCommGroup(group_name, ranks);
}

int GetRankIDByGroup(const std::string &group_name) { return MPIWrapper::instance().GetRankIDByGroup(group_name); }

int GetGroupSize(const std::string &group_name) { return MPIWrapper::instance().GetGroupSize(group_name); }

bool DestroyGroup(const std::string &group_name) { return MPIWrapper::instance().DestroyGroup(group_name); }

ncclResult_t AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                       ncclRedOp_t reduce_type, cudaStream_t stream, const std::string &group) {
  return NCCLWrapper::instance().AllReduce(input_addr, output_addr, count, data_type, reduce_type, stream, group);
}

ncclResult_t AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                       cudaStream_t stream, const std::string &group) {
  return NCCLWrapper::instance().AllGather(input_addr, output_addr, count, data_type, stream, group);
}

ncclResult_t ReduceScatter(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                           ncclRedOp_t reduce_type, cudaStream_t stream, const std::string &group) {
  return NCCLWrapper::instance().ReduceScatter(input_addr, output_addr, count, data_type, reduce_type, stream, group);
}

ncclResult_t Broadcast(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type, int root,
                       cudaStream_t stream, const std::string &group) {
  return NCCLWrapper::instance().Broadcast(input_addr, output_addr, count, data_type, root, stream, group);
}

ncclResult_t Send(const void *send_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
                  const std::string &group_name) {
  return NCCLWrapper::instance().Send(send_addr, count, data_type, peer_rank, stream, group_name);
}

ncclResult_t Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank, cudaStream_t stream,
                  const std::string &group_name) {
  return NCCLWrapper::instance().Recv(recv_addr, count, data_type, peer_rank, stream, group_name);
}

ncclResult_t GroupStart() { return NCCLWrapper::instance().GroupStart(); }

ncclResult_t GroupEnd() { return NCCLWrapper::instance().GroupEnd(); }

std::vector<int> GetGroupRanks(const std::string &group_name) {
  return NCCLWrapper::instance().GetGroupRanks(group_name);
}
