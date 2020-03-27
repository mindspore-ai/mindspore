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

#include "parallel/group_manager.h"

#include <vector>
#include <algorithm>

#include "parallel/device_manager.h"
#include "parallel/ops_info/ops_utils.h"
#include "utils/comm_manager.h"

namespace mindspore {
namespace parallel {
Group::Group() {
  name_.clear();
  devices_.clear();
}

Status Group::Init(const std::string &name, const std::list<Device> &devices) {
  this->name_ = name;
  this->devices_ = devices;
  return Status::SUCCESS;
}

std::list<Device> Group::GetDevicesList() const { return devices_; }

bool Group::IsInThisGroup(int32_t device_rank) {
  for (auto &device : devices_) {
    if (device.rank() == device_rank) {
      return true;
    }
  }
  return false;
}

// Get the position of the device in the group
Status Group::GetIndex(size_t *index) {
  size_t pos = 0;
  CheckGlobalDeviceManager();
  int32_t rank = g_device_manager->global_rank();
  for (auto &device : devices_) {
    if (device.rank() == rank) {
      *index = pos;
      return Status::SUCCESS;
    } else {
      pos++;
    }
  }
  MS_LOG(ERROR) << "Could not find device rank " << rank << "in this group!";
  return Status::FAILED;
}

GroupManager::GroupManager() { groups_.clear(); }

Status GroupManager::CreateGroup(const std::string &group_name, const std::list<Device> &devices,
                                 mindspore::parallel::Group *const group) {
  // it is simple to use size to determine whether it is a world group
  uint32_t world_size = 0;
  if (world_group_ != NCCL_WORLD_GROUP) {
    (void)CommManager::GetInstance().GetRankSize(world_group_, &world_size);
  }

  if ((world_group_ == NCCL_WORLD_GROUP) || (devices.size() == world_size)) {
    auto it = groups_.find(world_group_);
    if (it == groups_.end()) {
      (void)group->Init(world_group_, devices);
      groups_[world_group_] = *group;
    } else {
      *group = it->second;
    }
    MS_LOG(INFO) << "It is world group " << world_group_ << ", no need to create it.";
    return Status::SUCCESS;
  }

  auto it = groups_.find(group_name);
  // If there already exits a group with the desired 'name',
  // let the pointer point to the group.
  if (it != groups_.end()) {
    *group = it->second;
    return Status::SUCCESS;
  } else {
    (void)group->Init(group_name, devices);
    groups_[group_name] = *group;

    vector<uint32_t> ranks;
    (void)std::transform(std::begin(devices), std::end(devices), std::back_inserter(ranks),
                         [](const Device dev) { return (uint32_t)dev.rank(); });
    // Create group through the CommManager interface
    bool ret = CommManager::GetInstance().CreateGroupSync(group_name, ranks);
    if (!ret) {
      MS_LOG(ERROR) << "Create group failed, group name is " << group_name;
      return Status::FAILED;
    }

    MS_LOG(INFO) << "Create group success, group name is " << group_name;
    return Status::SUCCESS;
  }
}

Status GroupManager::DestroyGroup(mindspore::parallel::Group *const group) {
  std::string name = (*group).name();
  auto it = groups_.find(name);
  if (it == groups_.end()) {
    MS_LOG(ERROR) << "Could not find group name :" << name;
    return Status::FAILED;
  }
  (void)groups_.erase(it);
  bool ret = CommManager::GetInstance().DestroyGroup(name);
  if (!ret) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

Status GroupManager::DestroyAllGroups() {
  for (auto &it : groups_) {
    std::string name = it.first;
    bool ret = CommManager::GetInstance().DestroyGroup(name);
    if (!ret) {
      return Status::FAILED;
    }
  }
  groups_.clear();
  return Status::SUCCESS;
}

Status GroupManager::GetRankID(const std::string &name, unsigned int *const rank_id) {
  auto it = groups_.find(name);
  if (it == groups_.end()) {
    MS_LOG(ERROR) << "Could not find group name :" << name;
    return Status::FAILED;
  }
  bool ret = CommManager::GetInstance().GetRankID(name, rank_id);
  if (!ret) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

Status GroupManager::GetRankSize(const std::string &name, unsigned int *const rank_size) {
  auto it = groups_.find(name);
  if (it == groups_.end()) {
    MS_LOG(ERROR) << "Could not find group name :" << name;
    return Status::FAILED;
  }
  bool ret = CommManager::GetInstance().GetRankSize(name, rank_size);
  if (!ret) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

Status GroupManager::FindGroup(const std::string &name, mindspore::parallel::Group **group) {
  auto it = groups_.find(name);
  if (it == groups_.end()) {
    return Status::FAILED;
  }
  *group = &it->second;
  return Status::SUCCESS;
}

void GroupManager::Clear() { (void)DestroyAllGroups(); }
}  // namespace parallel
}  // namespace mindspore
