/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/lowlatency_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace ascend {
LowlatencyCollectiveCommLib::LowlatencyCollectiveCommLib() { global_group_name_ = kLCCLGlobalGroupName; }

bool LowlatencyCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool LowlatencyCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                           const std::vector<uint32_t> &group_ranks,
                                                           uint32_t local_group_rank, uint32_t local_group_size) {
  CHECK_RET((groups_.count(group_name) == 0), true, "The LCCL group " + group_name + " has already existed.");

  LowlatencyCommunicationGroupPtr group = std::make_shared<LowlatencyCommunicationGroup>(
    group_name, group_ranks, global_rank_id_, local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

int LowlatencyCollectiveCommLib::AllReduce(void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
                                           const HcclReduceOp reduce_op, const std::string &group_name,
                                           const aclrtStream stream) {
  std::cout << "allreduce lccl" << std::endl;
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator()->AllReduce(send_buff, recv_buff, count, data_type, reduce_op, stream);
}

int LowlatencyCollectiveCommLib::AllGather(void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
                                           const std::string &group_name, const aclrtStream stream) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator()->AllGather(send_buff, recv_buff, count, data_type, stream);
}

int LowlatencyCollectiveCommLib::ReduceScatter(void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
                                               const HcclReduceOp reduce_op, const std::string &group_name,
                                               const aclrtStream stream) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator()->ReduceScatter(send_buff, recv_buff, count, data_type, reduce_op, stream);
}

int LowlatencyCollectiveCommLib::All2All(void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
                                         const std::string &group_name, const aclrtStream stream) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator()->All2All(send_buff, recv_buff, count, data_type, stream);
}

int LowlatencyCollectiveCommLib::Broadcast(void *buff, size_t count, HcclDataType data_type, int root,
                                           const std::string &group_name, const aclrtStream stream) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator()->Broadcast(buff, count, data_type, root, stream);
}

LcclPtr LowlatencyCollectiveCommLib::LcclCommunicator(const std::string &group_name) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator();
}
}  // namespace ascend

using LowlatencyCollectiveCommLib = mindspore::device::ascend::LowlatencyCollectiveCommLib;

CollectiveCommunicationLib *communication_lib_instance() { return &LowlatencyCollectiveCommLib::GetInstance(); }
}  // namespace device
}  // namespace mindspore
