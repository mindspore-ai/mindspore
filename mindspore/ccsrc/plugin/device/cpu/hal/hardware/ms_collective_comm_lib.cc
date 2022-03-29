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
  finalized_ = false;
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

bool MsCollectiveCommLib::AllGatherHostHashName(size_t host_hash_name, std::vector<size_t> *host_hash_names) const {
  CHECK_IF_NULL(host_hash_names);
  while (!SendHostHashName(host_hash_name)) {
    MS_LOG(WARNING) << "Send host hash name to scheduler failed, retrying...";
    if (finalized_.load()) {
      return false;
    }

    std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
  }

  while (!QueryHostHashNames(host_hash_names)) {
    MS_LOG(WARNING) << "Query host hash names from scheduler failed, retrying...";
    if (finalized_.load()) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
  }

  return true;
}

bool MsCollectiveCommLib::BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) {
  CHECK_IF_NULL(root_info);
  auto group = GetGroup(group_name);
  CHECK_IF_NULL(group);
  uint32_t group_rank_id = group->GetGroupRank(node_->rank_id());
  if (group_rank_id == 0) {
    while (!SendUniqueID(group_name, root_info_size, root_info)) {
      MS_LOG(WARNING) << "Send unique id to scheduler failed, retrying...";
      if (finalized_.load()) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
    }
    return true;
  }

  while (!QueryUniqueID(group_name, root_info_size, root_info)) {
    MS_LOG(WARNING) << "Query unique id from scheduler failed, retrying...";
    if (finalized_.load()) {
      return false;
    }

    std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
  }
  return true;
}

bool MsCollectiveCommLib::SendHostHashName(size_t host_hash_name) const {
  CHECK_IF_NULL(node_);
  ps::core::SendHostHashNameMessage send_host_name_msg;
  send_host_name_msg.set_node_id(node_->node_id());
  send_host_name_msg.set_rank_id(node_->rank_id());
  send_host_name_msg.set_host_hash_name(host_hash_name);
  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  if (!node_->SendToScheduler(send_host_name_msg.SerializeAsString().data(),
                              send_host_name_msg.SerializeAsString().size(), NodeCommand::SEND_HOST_NAME, &output)) {
    MS_LOG(WARNING) << "Failed to send host hash name request to scheduler.";
    return false;
  }

  ps::core::GeneralResponseMsg resp_msg;
  CHECK_IF_NULL(output);
  (void)resp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
  if (!resp_msg.is_success()) {
    MS_LOG(WARNING) << "Send host hash name to scheduler failed.";
    return false;
  }
  return true;
}

bool MsCollectiveCommLib::QueryHostHashNames(std::vector<size_t> *host_hash_names) const {
  CHECK_IF_NULL(host_hash_names);
  CHECK_IF_NULL(node_);
  ps::core::GeneralQueryMessage general_query_msg;
  general_query_msg.set_node_id(node_->node_id());
  general_query_msg.set_rank_id(node_->rank_id());
  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  if (!node_->SendToScheduler(general_query_msg.SerializeAsString().data(),
                              general_query_msg.SerializeAsString().size(), NodeCommand::QUERY_HOST_NAMES, &output)) {
    MS_LOG(WARNING) << "Failed to send query host name request to scheduler.";
    return false;
  }

  ps::core::QueryHostHashNameRespMessage resp_msg;
  CHECK_IF_NULL(output);
  (void)resp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
  if (!resp_msg.is_success()) {
    MS_LOG(INFO) << "Query host hash name from scheduer failed, maybe scheduler has not received all host names.";
    return false;
  }

  if (host_hash_names->size() != IntToSize(resp_msg.host_hash_names_size())) {
    MS_LOG(ERROR) << "The host_hash_names container size: " << host_hash_names->size()
                  << ", but received size: " << resp_msg.host_hash_names_size();
    return false;
  }

  for (size_t i = 0; i < host_hash_names->size(); i++) {
    (*host_hash_names)[i] = resp_msg.host_hash_names()[i];
  }

  return true;
}

bool MsCollectiveCommLib::SendUniqueID(const std::string &group_name, size_t root_info_size,
                                       const void *root_info) const {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(node_);

  ps::core::SendUniqueIDMessage send_unique_id_msg;
  send_unique_id_msg.set_node_id(node_->node_id());
  send_unique_id_msg.set_rank_id(0);
  send_unique_id_msg.set_group_name(group_name);
  send_unique_id_msg.set_unique_id(root_info, root_info_size);

  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  if (!node_->SendToScheduler(send_unique_id_msg.SerializeAsString().data(),
                              send_unique_id_msg.SerializeAsString().size(), NodeCommand::SEND_UNIQUE_ID, &output)) {
    MS_LOG(WARNING) << "Failed to send unique id request to scheduler.";
    return false;
  }

  ps::core::GeneralResponseMsg resp_msg;
  CHECK_IF_NULL(output);
  (void)resp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
  if (!resp_msg.is_success()) {
    MS_LOG(WARNING) << "Send unique id to scheduler failed.";
    return false;
  }
  return true;
}

bool MsCollectiveCommLib::QueryUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) const {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(node_);
  ps::core::QueryUniqueIDMessage query_unique_id_msg;
  query_unique_id_msg.set_node_id(node_->node_id());
  query_unique_id_msg.set_group_name(group_name);
  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  if (!node_->SendToScheduler(query_unique_id_msg.SerializeAsString().data(),
                              query_unique_id_msg.SerializeAsString().size(), NodeCommand::QUERY_UNIQUE_ID, &output)) {
    MS_LOG(WARNING) << "Failed to send query unique id request to scheduler.";
    return false;
  }

  ps::core::QueryUniqueIDRespMessage resp_msg;
  CHECK_IF_NULL(output);
  (void)resp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
  if (!resp_msg.is_success()) {
    MS_LOG(INFO) << "Query unique id from scheduer failed, maybe scheduler has not received unique id.";
    return false;
  }

  auto ret = memcpy_s(root_info, root_info_size, resp_msg.unique_id().data(), resp_msg.unique_id().length());
  if (ret != EOK) {
    MS_LOG(WARNING) << "The memcpy_s error, errorno(" << ret << ")";
    return false;
  }
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
