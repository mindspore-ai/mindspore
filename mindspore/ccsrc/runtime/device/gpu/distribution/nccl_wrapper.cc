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

#include "runtime/device/gpu/distribution/nccl_wrapper.h"

namespace mindspore {
namespace device {
namespace gpu {
NCCLWrapper &NCCLWrapper::instance() {
  static NCCLWrapper instance;
  return instance;
}

ncclUniqueId NCCLWrapper::nccl_unique_id() const {
  ncclUniqueId unique_id;
  CHECK_RET(ncclGetUniqueId(&unique_id), ncclSuccess, "Failed to create nccl unique id.");
  return unique_id;
}

void NCCLWrapper::set_nccl_unique_id(ncclUniqueId unique_id) { unique_id_ = unique_id; }

void NCCLWrapper::set_rank(int rank_id, int rank_size) {
  rank_id_ = rank_id;
  rank_size_ = rank_size;
}

void NCCLWrapper::InitNCCLComm() {
  CHECK_RET(ncclCommInitRank(&comm_, rank_size_, unique_id_, rank_id_), ncclSuccess,
            "Failed to init nccl communicator.");
  group_to_comm_map_[NCCL_WORLD_GROUP] = comm_;
}

void NCCLWrapper::InitNCCLComm(ncclComm_t *comm, int rank_size, ncclUniqueId unique_id, int rank) {
  CHECK_RET(ncclCommInitRank(comm, rank_size, unique_id, rank), ncclSuccess, "Failed to init nccl communicator.");
}

ncclResult_t NCCLWrapper::AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    ncclRedOp_t reduce_type, cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_to_comm_map_.count(group_name), 1,
            "Failed to find NCCL communicator for AllReduce by the group name " + group_name);
  ncclComm_t group_comm = group_to_comm_map_[group_name];
  return ncclAllReduce(input_addr, output_addr, count, data_type, reduce_type, group_comm, stream);
}

ncclResult_t NCCLWrapper::AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_to_comm_map_.count(group_name), 1,
            "Failed to find NCCL communicator for AllGather by the group name " + group_name);
  ncclComm_t group_comm = group_to_comm_map_[group_name];
  return ncclAllGather(input_addr, output_addr, count, data_type, group_comm, stream);
}

ncclResult_t NCCLWrapper::ReduceScatter(const void *input_addr, void *output_addr, size_t count,
                                        ncclDataType_t data_type, ncclRedOp_t reduce_type, cudaStream_t stream,
                                        const std::string &group_name) {
  CHECK_RET(group_to_comm_map_.count(group_name), 1,
            "Failed to find NCCL communicator for ReduceScatter by the group name " + group_name);
  ncclComm_t group_comm = group_to_comm_map_[group_name];
  return ncclReduceScatter(input_addr, output_addr, count, data_type, reduce_type, group_comm, stream);
}

void NCCLWrapper::SetGroupNameToNCCLComm(const std::string &group_name, const ncclComm_t comm) {
  group_to_comm_map_[group_name] = comm;
}

void NCCLWrapper::DestroyGroup(const std::string &group_name) {
  auto group_iter = group_to_comm_map_.find(group_name);
  if (group_iter == group_to_comm_map_.end()) {
    return;
  }
  group_to_comm_map_.erase(group_iter);
  ncclComm_t group_comm = group_iter->second;
  CHECK_RET(ncclCommDestroy(group_comm), ncclSuccess, "Failed to destroy NCCL communicator for " + group_name);
  return;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
