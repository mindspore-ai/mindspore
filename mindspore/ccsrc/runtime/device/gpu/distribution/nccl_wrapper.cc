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

#include <vector>
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

void NCCLWrapper::InitNCCLComm() {
  if (comm_init_done_) {
    return;
  }

  for (auto group : group_info_) {
    std::string group_name = group.first;
    NcclGroupInfo group_info = group.second;
    CHECK_RET(ncclCommInitRank(&(group_info.comm), group_info.size, group_info.unique_id, group_info.rank), ncclSuccess,
              "Failed to init nccl communicator for group " + group_name);
    group_info_[group_name].comm = group_info.comm;
  }
  comm_init_done_ = true;
}

ncclResult_t NCCLWrapper::AllReduce(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    ncclRedOp_t reduce_type, cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1,
            "Failed to find NCCL communicator for AllReduce by the group name " + group_name);
  ncclComm_t group_comm = group_info_[group_name].comm;
  return ncclAllReduce(input_addr, output_addr, count, data_type, reduce_type, group_comm, stream);
}

ncclResult_t NCCLWrapper::AllGather(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1,
            "Failed to find NCCL communicator for AllGather by the group name " + group_name);
  ncclComm_t group_comm = group_info_[group_name].comm;
  return ncclAllGather(input_addr, output_addr, count, data_type, group_comm, stream);
}

ncclResult_t NCCLWrapper::ReduceScatter(const void *input_addr, void *output_addr, size_t count,
                                        ncclDataType_t data_type, ncclRedOp_t reduce_type, cudaStream_t stream,
                                        const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1,
            "Failed to find NCCL communicator for ReduceScatter by the group name " + group_name);
  ncclComm_t group_comm = group_info_[group_name].comm;
  return ncclReduceScatter(input_addr, output_addr, count, data_type, reduce_type, group_comm, stream);
}

ncclResult_t NCCLWrapper::Broadcast(const void *input_addr, void *output_addr, size_t count, ncclDataType_t data_type,
                                    int root, cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1,
            "Failed to find NCCL communicator for Broadcast by the group name " + group_name);
  ncclComm_t group_comm = group_info_[group_name].comm;
  return ncclBroadcast(input_addr, output_addr, count, data_type, root, group_comm, stream);
}

ncclResult_t NCCLWrapper::Send(const void *send_addr, size_t count, ncclDataType_t data_type, int peer_rank,
                               cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1, "Failed to find group info for Send by the group name " + group_name);
  ncclComm_t group_comm = group_info_[group_name].comm;
  return ncclSend(send_addr, count, data_type, peer_rank, group_comm, stream);
}

ncclResult_t NCCLWrapper::Recv(void *recv_addr, size_t count, ncclDataType_t data_type, int peer_rank,
                               cudaStream_t stream, const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1, "Failed to find group info for Recv by the group name " + group_name);
  ncclComm_t group_comm = group_info_[group_name].comm;
  return ncclRecv(recv_addr, count, data_type, peer_rank, group_comm, stream);
}

ncclResult_t NCCLWrapper::GroupStart() { return ncclGroupStart(); }

ncclResult_t NCCLWrapper::GroupEnd() { return ncclGroupEnd(); }

void NCCLWrapper::AddGroupInfo(const std::string &group_name, NcclGroupInfo *group) {
  if (comm_init_done_) {
    CHECK_RET(ncclCommInitRank(&(group->comm), group->size, group->unique_id, group->rank), ncclSuccess,
              "Failed to init nccl communicator for group " + group_name);
  }
  group_info_[group_name] = *group;
}

void NCCLWrapper::DestroyGroup(const std::string &group_name) {
  auto group_iter = group_info_.find(group_name);
  if (group_iter == group_info_.end()) {
    return;
  }
  ncclComm_t group_comm = group_iter->second.comm;
  CHECK_RET(ncclCommDestroy(group_comm), ncclSuccess, "Failed to destroy NCCL communicator for " + group_name);
  group_info_.erase(group_iter);
  return;
}

std::vector<int> NCCLWrapper::GetGroupRanks(const std::string &group_name) {
  CHECK_RET(group_info_.count(group_name), 1,
            "Failed to find group info for GetGroupRanks by the group name " + group_name);
  return group_info_[group_name].group_ranks;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
