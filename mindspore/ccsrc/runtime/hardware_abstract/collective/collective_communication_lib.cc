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

#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"

namespace mindspore {
namespace device {
bool CollectiveCommunicationLib::Finalize() {
  if (!initialized_ || finalized_.load()) {
    return true;
  }

  for (const auto &group : groups_) {
    CHECK_IF_NULL(group.second);
    if (!group.second->Finalize()) {
      return false;
    }
  }
  groups_.clear();
  initialized_ = false;
  finalized_ = true;
  return true;
}

bool CollectiveCommunicationLib::DestroyCommunicationGroup(const std::string &group_name) {
  if (groups_.count(group_name) == 0) {
    return false;
  }
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  if (!group->Finalize()) {
    return false;
  }
  (void)groups_.erase(group_name);
  return true;
}

uint32_t CollectiveCommunicationLib::GetRankId(const std::string &group_name) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->GetGroupRank(global_rank_id_);
}

uint32_t CollectiveCommunicationLib::GetGroupSize(const std::string &group_name) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->group_size();
}

uint32_t CollectiveCommunicationLib::GetLocalRankId(const std::string &group_name) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->GetLocalGroupRank();
}

uint32_t CollectiveCommunicationLib::GetLocalGroupSize(const std::string &group_name) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->local_group_size();
}

uint32_t CollectiveCommunicationLib::GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->GetGlobalRank(local_rank);
}

uint32_t CollectiveCommunicationLib::GetGroupRankFromWorldRank(uint32_t global_rank, const std::string &group_name) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->GetGroupRank(global_rank);
}

CommunicationGroupPtr CollectiveCommunicationLib::GetGroup(const std::string &group_name) {
  if (groups_.count(group_name) == 0) {
    return nullptr;
  }
  return groups_[group_name];
}

void CollectiveCommunicationLib::SetLocalGroupRank(const std::string &group_name, uint32_t local_rank_id) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  group->set_local_rank(local_rank_id);
}

void CollectiveCommunicationLib::SetLocalGroupSize(const std::string &group_name, uint32_t local_group_size) {
  CHECK_RET(groups_.count(group_name) != 0, true, "The group " + group_name + " does not exist.");
  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  group->set_local_size(local_group_size);
}

const std::string &CollectiveCommunicationLib::global_group_name() const { return global_group_name_; }

uint32_t CollectiveCommunicationLib::global_rank_id() const { return global_rank_id_; }

uint32_t CollectiveCommunicationLib::local_rank_id() const { return local_rank_id_; }

uint32_t CollectiveCommunicationLib::global_rank_size() const { return global_rank_size_; }

void CollectiveCommunicationLib::SetHelperCommLib(CollectiveCommunicationLib *comm_lib) {}
}  // namespace device
}  // namespace mindspore
