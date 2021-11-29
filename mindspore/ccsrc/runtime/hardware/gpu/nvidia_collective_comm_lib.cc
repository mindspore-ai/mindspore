/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/hardware/gpu/nvidia_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace gpu {
NvidiaCollectiveCommLib::NvidiaCollectiveCommLib() { global_group_name_ = kNCCLGlobalGroupName; }

bool NvidiaCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  initialized_ = true;
  return true;
}

bool NvidiaCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                       const std::vector<uint32_t> &group_ranks) {
  CHECK_RET((groups_.count(group_name) == 0), true, "The NCCL group " + group_name + " has already existed.");

  NvidiaCommunicationGroupPtr group =
    std::make_shared<NvidiaCommunicationGroup>(group_name, group_ranks, global_rank_id_);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

bool NvidiaCollectiveCommLib::AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                        const std::string &group_name, void *stream) {
  if (!CheckNCCLDataType(data_type)) {
    return false;
  }

  CHECK_RET((groups_.count(group_name) != 0), true, "The NCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<NvidiaCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  CHECK_RET(ncclAllGather(send_buff, recv_buff, send_count, kNCCLDataTypeMap.at(data_type), group->nccl_communicator(),
                          static_cast<cudaStream_t>(stream)),
            ncclSuccess, "ncclAllGather failed.");
  return true;
}

bool NvidiaCollectiveCommLib::AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                        ReduceMode reduce_op, const std::string &group_name, void *stream) {
  if (!CheckNCCLDataType(data_type)) {
    return false;
  }
  if (!CheckNCCLReduceType(reduce_op)) {
    return false;
  }

  CHECK_RET((groups_.count(group_name) != 0), true, "The NCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<NvidiaCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  CHECK_RET(
    ncclAllReduce(send_buff, recv_buff, send_count, kNCCLDataTypeMap.at(data_type), kNCCLReduceTypeMap.at(reduce_op),
                  group->nccl_communicator(), static_cast<cudaStream_t>(stream)),
    ncclSuccess, "ncclAllReduce failed.");
  return true;
}

bool NvidiaCollectiveCommLib::Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                        uint32_t root_rank, const std::string &group_name, void *stream) {
  if (!CheckNCCLDataType(data_type)) {
    return false;
  }

  CHECK_RET((groups_.count(group_name) != 0), true, "The NCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<NvidiaCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  CHECK_RET(ncclBroadcast(send_buff, recv_buff, send_count, kNCCLDataTypeMap.at(data_type), root_rank,
                          group->nccl_communicator(), static_cast<cudaStream_t>(stream)),
            ncclSuccess, "ncclBroadcast failed.");
  return true;
}

bool NvidiaCollectiveCommLib::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                                            ReduceMode reduce_op, const std::string &group_name, void *stream) {
  if (!CheckNCCLDataType(data_type)) {
    return false;
  }
  if (!CheckNCCLReduceType(reduce_op)) {
    return false;
  }

  CHECK_RET((groups_.count(group_name) != 0), true, "The NCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<NvidiaCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  CHECK_RET(
    ncclReduceScatter(send_buff, recv_buff, recv_count, kNCCLDataTypeMap.at(data_type),
                      kNCCLReduceTypeMap.at(reduce_op), group->nccl_communicator(), static_cast<cudaStream_t>(stream)),
    ncclSuccess, "ncclReduceScatter failed.");
  return true;
}

bool NvidiaCollectiveCommLib::Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer,
                                   const std::string &group_name, void *stream) {
  if (!CheckNCCLDataType(data_type)) {
    return false;
  }

  CHECK_RET((groups_.count(group_name) != 0), true, "The NCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<NvidiaCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  CHECK_RET(ncclSend(send_buff, count, kNCCLDataTypeMap.at(data_type), peer, group->nccl_communicator(),
                     static_cast<cudaStream_t>(stream)),
            ncclSuccess, "ncclSend failed.");
  return true;
}

bool NvidiaCollectiveCommLib::Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer,
                                   const std::string &group_name, void *stream) {
  if (!CheckNCCLDataType(data_type)) {
    return false;
  }

  CHECK_RET((groups_.count(group_name) != 0), true, "The NCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<NvidiaCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  CHECK_RET(ncclRecv(recv_buff, count, kNCCLDataTypeMap.at(data_type), peer, group->nccl_communicator(),
                     static_cast<cudaStream_t>(stream)),
            ncclSuccess, "ncclRecv failed.");
  return true;
}

bool NvidiaCollectiveCommLib::CheckNCCLDataType(TypeId data_type) {
  CHECK_RET((kNCCLDataTypeMap.count(data_type) != 0), true,
            "Data type " + std::to_string(data_type) + " is not supported in NCCL.");
  return true;
}

bool NvidiaCollectiveCommLib::CheckNCCLReduceType(ReduceMode reduce_op) {
  CHECK_RET((kNCCLReduceTypeMap.count(reduce_op) != 0), true,
            "Reduce type " + std::to_string(reduce_op) + " is not supported in NCCL.");
  return true;
}
}  // namespace gpu

using NvidiaCollectiveCommLib = mindspore::device::gpu::NvidiaCollectiveCommLib;
CollectiveCommunicationLib *communication_lib_instance() { return &NvidiaCollectiveCommLib::GetInstance(); }
}  // namespace device
}  // namespace mindspore
