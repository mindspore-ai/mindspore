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

#include "plugin/device/cpu/hal/hardware/ms_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace cpu {
MsCollectiveCommLib::MsCollectiveCommLib() {
  node_ = std::dynamic_pointer_cast<ps::core::AbstractNode>(ClusterContext::instance()->node());
  // Generate the global group name with node role.
  global_group_name_ = ClusterContext::instance()->node_role() + "_" + kMSGlobalGroupName;
  MS_LOG(INFO) << "Global group name of MindSpore collective communication library is " << global_group_name_;
}

bool MsCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  initialized_ = true;
  return true;
}

bool MsCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                   const std::vector<uint32_t> &group_ranks) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(ERROR) << "The group " << group_name << " has already existed.";
    return false;
  }

  MsCommunicationGroupPtr group = std::make_shared<MsCommunicationGroup>(group_name, group_ranks, global_rank_id_);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

bool MsCollectiveCommLib::AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                    const std::string &, void *) {
  CHECK_IF_NULL(send_buff);
  CHECK_IF_NULL(recv_buff);
  CHECK_IF_NULL(node_);

  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().AllGather<char>(send_buff, recv_buff, send_count, node_);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().AllGather<int32_t>(send_buff, recv_buff, send_count, node_);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().AllGather<uint64_t>(send_buff, recv_buff, send_count, node_);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().AllGather<float>(send_buff, recv_buff, send_count, node_);
    default:
      return false;
  }
  return true;
}

bool MsCollectiveCommLib::Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                    uint32_t root_rank, const std::string &group_name, void *) {
  CHECK_IF_NULL(send_buff);
  CHECK_IF_NULL(recv_buff);
  CHECK_IF_NULL(node_);

  if (groups_.count(group_name) == 0) {
    MS_LOG(ERROR) << "The group " << group_name << " does not exist.";
    return false;
  }

  auto group = groups_[group_name];
  CommunicationGroupInfo group_info = {};
  group_info.size = group->group_size();
  group_info.global_rank = global_rank_id_;
  group_info.group_ranks = group->group_ranks();
  group_info.global_to_group_ranks = group->global_to_group_ranks();
  group_info.group_to_global_ranks = group->group_to_global_ranks();

  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().Broadcast<char>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Broadcast<int32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                 group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Broadcast<uint64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                  group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Broadcast<float>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    default:
      return false;
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
