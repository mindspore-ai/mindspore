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

#include "include/backend/distributed/constants.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "runtime/collective/collective_communication_lib.h"
#include "plugin/device/cpu/hal/hardware/allreduce_impl.h"

namespace mindspore {
namespace device {
namespace cpu {
using distributed::recovery::RecoveryContext;

// These keywords is used for synchronization of collective communication's metadata(eg. unique id).
constexpr char kGroupInfoPrefix[] = "group_info_";
constexpr char kGroupName[] = "group_name";
constexpr char kUniqueId[] = "unique_id";
MsCollectiveCommLib::MsCollectiveCommLib() {
  // Generate the global group name with node role.
  global_group_name_ = kMCCLGlobalGroupName;
  MS_LOG(INFO) << "Global group name of MindSpore collective communication library is " << global_group_name_;
}

bool MsCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    MS_LOG(WARNING) << "MsCollectiveCommLib has already been initialized.";
    return true;
  }

  launcher_ = std::make_unique<AllReduceLauncher>();
  CHECK_IF_NULL(launcher_);
  if (!launcher_->Initialize()) {
    MS_LOG(EXCEPTION) << "Failed to initialize the allreduce launcher.";
  }
  node_ = launcher_->collective_node();

  cgn_ = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    ClusterContext::instance()->node_base());

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool MsCollectiveCommLib::Finalize() {
  if (launcher_ != nullptr) {
    return launcher_->Finalize();
  }
  return true;
}

bool MsCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                   const std::vector<uint32_t> &group_ranks, uint32_t local_group_rank,
                                                   uint32_t local_group_size) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(WARNING) << "The group " << group_name << " has already existed.";
    return true;
  }

  MsCommunicationGroupPtr group = std::make_shared<MsCommunicationGroup>(group_name, group_ranks, global_rank_id_,
                                                                         local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

bool MsCollectiveCommLib::AllGatherHostHashName(size_t host_hash_name, std::vector<size_t> *host_hash_names) const {
  CHECK_IF_NULL(host_hash_names);
  CHECK_IF_NULL(cgn_);

  auto role = common::GetEnv(distributed::kEnvRole);
  bool success = false;
  // It this is not recovery scenario, retry for 3*80s, which is 4 minutes.
  const size_t interval = 3;
  size_t retry = RecoveryContext::GetInstance()->enable_recovery() ? SIZE_MAX : kMSCollectiveRetryTime;
  while (!success && --retry > 0) {
    auto hostnames = cgn_->GetHostNames(role);
    if (hostnames.size() < host_hash_names->size()) {
      (void)sleep(interval);
      continue;
    } else if (hostnames.size() > host_hash_names->size()) {
      MS_LOG(ERROR) << "Invalid number of hostnames, expected number of hostnames: " << host_hash_names->size()
                    << ", actual number of hostnames: " << hostnames.size();
      return false;
    }

    for (size_t i = 0; i < host_hash_names->size(); i++) {
      size_t host_hash = std::hash<std::string>()(hostnames[i]);
      (*host_hash_names)[i] = host_hash;
    }
    success = true;
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Failed to AllGather host's hash name due to timeout.";
  }

  return true;
}

bool MsCollectiveCommLib::BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(node_);
  CHECK_IF_NULL(cgn_);
  auto group = GetGroup(group_name);
  CHECK_IF_NULL(group);

  if (!synchronized_) {
    node_->SynchronizeAddresses();
  } else {
    synchronized_ = false;
  }

  uint32_t group_rank_id = group->GetGroupRank(cgn_->rank_id());
  if (group_rank_id == 0) {
    while (!SendUniqueID(group_name, root_info_size, root_info)) {
      MS_LOG(WARNING) << "Send unique id to scheduler failed, retrying...";
      if (finalized_.load()) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
    }
  } else {
    while (!QueryUniqueID(group_name, root_info_size, root_info)) {
      MS_LOG(WARNING) << "Query unique id from scheduler failed, retrying...";
      if (finalized_.load()) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
    }
  }
  return true;
}

bool MsCollectiveCommLib::SendUniqueID(const std::string &group_name, size_t root_info_size,
                                       const void *root_info) const {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(node_);
  CHECK_IF_NULL(cgn_);

  // Create the group info which contains the unique id and send it to the meta server.
  std::string node_role_prefix = cgn_->role() + "_";
  std::string group_info_key = node_role_prefix + kGroupInfoPrefix + group_name;

  bool success = false;
  // It this is not recovery scenario, retry for 3*80s, which is 4 minutes.
  const size_t interval = 3;
  size_t retry = RecoveryContext::GetInstance()->enable_recovery() ? SIZE_MAX : kMSCollectiveRetryTime;
  while (!success && --retry > 0) {
    success = cgn_->PutMetadata(group_info_key, root_info, root_info_size);
    if (!success) {
      MS_LOG(WARNING) << "Failed to send unique id for group " << group_name << ". Retry time " << retry;
      (void)sleep(interval);
    }
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Failed to send unique id to the meta server node due to timeout.";
  }
  MS_LOG(INFO) << "The unique id for group " << group_name << " has been registered to the meta server.";
  return true;
}

bool MsCollectiveCommLib::QueryUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) const {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(node_);
  CHECK_IF_NULL(cgn_);

  std::string node_role_prefix = cgn_->role() + "_";
  std::string group_info_key = node_role_prefix + kGroupInfoPrefix + group_name;

  bool success = false;
  // It this is not recovery scenario, retry for 3*80s, which is 4 minutes.
  const size_t interval = 3;
  size_t retry = RecoveryContext::GetInstance()->enable_recovery() ? SIZE_MAX : kMSCollectiveRetryTime;
  while (!success && --retry > 0) {
    auto unique_id = cgn_->GetMetadata(group_info_key);
    if (unique_id.length() > 0) {
      auto ret = memcpy_s(root_info, root_info_size, unique_id.data(), unique_id.length());
      if (ret != EOK) {
        MS_LOG(WARNING) << "The memcpy_s error, errorno(" << ret << ")";
        return false;
      }
      success = true;
    } else {
      MS_LOG(WARNING) << "Retry to lookup the unique id for group " << group_name << " from the meta server node...";
      (void)sleep(interval);
    }
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Failed to fetch the unique id of the collective lib from the meta server node.";
  }
  return true;
}

bool MsCollectiveCommLib::AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                    CollectiveOpReduceType reduce_op, const std::string &group_name, void *) {
  CHECK_IF_NULL(send_buff);
  CHECK_IF_NULL(recv_buff);
  CHECK_IF_NULL(launcher_);
  if (data_type != TypeId::kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "AllReduce only support float32.";
  }
  if (reduce_op != CollectiveOpReduceType::Reduce_Sum) {
    MS_LOG(EXCEPTION) << "AllReduce only support reduce sum.";
  }
  bool ret = launcher_->Execute(send_buff, recv_buff, send_count);
  return ret;
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
  CHECK_IF_NULL(group);
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
      [[fallthrough]];
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Broadcast<int32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                 group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Broadcast<uint64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                  group_info);
    case TypeId::kNumberTypeFloat32:
      [[fallthrough]];
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Broadcast<float>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    default:
      return false;
  }
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
