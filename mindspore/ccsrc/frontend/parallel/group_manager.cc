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

#include "frontend/parallel/group_manager.h"
#include <algorithm>
#include <vector>
#include <utility>
#if !defined(NO_DLIB) || defined(ENABLE_GPU)
#include "backend/session/executor_manager.h"
#else
#include "frontend/parallel/parallel_stub/executor_manager_stub.h"
#endif
#include "frontend/parallel/device_manager.h"
#include "utils/comm_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
Group::Group() {
  name_.clear();
  devices_.clear();
}

Status Group::Init(const std::string &name, const std::vector<Device> &devices) {
  this->name_ = name;
  this->devices_ = devices;
  return Status::SUCCESS;
}

std::vector<Device> Group::GetDevicesList() const { return devices_; }

bool Group::IsInThisGroup(int64_t device_rank) {
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
  int64_t rank = g_device_manager->global_rank();
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

#if !defined(NO_DLIB) || defined(ENABLE_GPU)
bool GroupManager::CreateGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                         const std::vector<uint32_t> ranks, int device_id) {
  auto executor = session::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  bool ret = executor->CreateCommGroup(group_name, ranks);
  return ret;
}

bool GroupManager::DestroyGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                          int device_id) {
  auto executor = session::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  bool ret = executor->DestroyCommGroup(group_name);
  return ret;
}

Status CreateGroups(const std::vector<std::pair<std::string, std::vector<uint32_t>>> &group_info) {
  // Create group through the executor
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto executor = session::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  for (auto &group : group_info) {
    bool ret = executor->CreateCommGroup(group.first, group.second);
    if (!ret) {
      MS_LOG(ERROR) << "Create group failed, group name is " << group.first << ", ranks is " << group.second;
      return FAILED;
    }
    MS_LOG(INFO) << "Create group success, group name is " << group.first << ", ranks is " << group.second;
  }

  return SUCCESS;
}
#else
bool GroupManager::CreateGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                         const std::vector<uint32_t> ranks, int device_id) {
  MS_LOG(WARNING) << "Create group in stub";
  auto executor = parallel::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  bool ret = executor->CreateCommGroup(group_name, ranks);
  return ret;
}

bool GroupManager::DestroyGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                          int device_id) {
  MS_LOG(WARNING) << "Destroy group in stub";
  auto executor = parallel::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  bool ret = executor->DestroyCommGroup(group_name);
  return ret;
}

Status CreateGroups(const std::vector<std::pair<std::string, std::vector<uint32_t>>> &group_info) {
  // Create group through the executor
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto executor = parallel::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  for (auto &group : group_info) {
    bool ret = executor->CreateCommGroup(group.first, group.second);
    if (!ret) {
      MS_LOG(ERROR) << "Create group failed, group name is " << group.first << ", ranks is " << group.second;
      return FAILED;
    }
    MS_LOG(INFO) << "Create group success, group name is " << group.first << ", ranks is " << group.second;
  }

  return SUCCESS;
}
#endif
Status GroupManager::CreateGroup(const std::string &group_name, const std::vector<Device> &devices,
                                 mindspore::parallel::Group *const group) {
  // it is simple to use size to determine whether it is a world group
  uint32_t world_size = 0;
  (void)CommManager::GetInstance().GetRankSize(world_group_, &world_size);

  if (devices.size() == world_size) {
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
    // Create group through the executor
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    bool ret = CreateGroupByExecutor(device_name, group_name, ranks, device_id);
    if (!ret) {
      MS_LOG(ERROR) << "Create group failed, group name is " << group_name;
      return Status::FAILED;
    }

    std::pair<std::string, std::vector<uint32_t>> group_info = std::make_pair(group_name, ranks);
    group_info_.push_back(group_info);

    MS_LOG(INFO) << "Create group success, group name is " << group_name;
    return Status::SUCCESS;
  }
}

Status GroupManager::DestroyGroup(const std::string &group_name) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  bool ret = DestroyGroupByExecutor(device_name, group_name, device_id);
  if (!ret) {
    return Status::FAILED;
  }
  return Status::SUCCESS;
}

Status GroupManager::DestroyGroup(mindspore::parallel::Group *const group) {
  std::string name = (*group).name();
  auto it = groups_.find(name);
  if (it == groups_.end()) {
    MS_LOG(ERROR) << "Could not find group name :" << name;
    return Status::FAILED;
  }
  (void)groups_.erase(it);
  return DestroyGroup(name);
}

Status GroupManager::DestroyAllGroups() {
  for (auto &it : groups_) {
    std::string name = it.first;
    auto ret = DestroyGroup(name);
    if (ret != Status::SUCCESS) {
      return Status::FAILED;
    }
  }
  groups_.clear();
  return Status::SUCCESS;
}

Status GroupManager::GetRankID(const std::string &name, uint32_t *const rank_id) {
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

Status GroupManager::GetRankSize(const std::string &name, uint32_t *const rank_size) {
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
