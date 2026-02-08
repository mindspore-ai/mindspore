/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#if (!defined(_WIN32) && !defined(__APPLE__) || defined(ENABLE_TEST))
#else
#include "frontend/parallel/parallel_stub/executor_manager_stub.h"
#endif
#include "frontend/parallel/device_manager.h"
#include "include/frontend/parallel/device_manager.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
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
  auto rank = ParallelContext::GetInstance()->global_rank();
  if (!ParallelContext::GetInstance()->do_transform()) {
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    rank = g_device_manager->global_rank();
  }
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

Status Group::GetIndexByRank(int64_t rank, size_t *index) {
  size_t pos = 0;
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

#if (!defined(_WIN32) && !defined(__APPLE__) || defined(ENABLE_TEST))
bool GroupManager::CreateGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                         const std::vector<uint32_t> ranks, uint32_t device_id) const {
  // The group operation thread must be same with nccl init thread in the GPU device.
  return CommManager::GetInstance().CreateGroupSync(group_name, ranks);
}

bool GroupManager::DestroyGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                          uint32_t device_id) const {
  // The group operation thread must be same with nccl init thread in the GPU device.
  return CommManager::GetInstance().DestroyGroup(group_name);
}

Status CreateGroups(const std::vector<std::pair<std::string, std::vector<uint32_t>>> &group_info) {
  // Create group through the executor
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto print_func = [](const auto &group, bool judge) {
    if (judge) {
      MS_LOG(INFO) << "Create group success, group name is " << group.first << ", ranks is " << group.second;
    } else {
      MS_LOG(ERROR) << "Create group failed, group name is " << group.first << ", ranks is " << group.second;
    }
  };
  // The group operation thread must be same with nccl init thread in the GPU device.
  for (auto &group : group_info) {
    bool ret = CommManager::GetInstance().CreateGroupSync(group.first, group.second);
    print_func(group, ret);
    if (!ret) {
      return FAILED;
    }
  }
  return SUCCESS;
}
#else
bool GroupManager::CreateGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                         const std::vector<uint32_t> ranks, uint32_t device_id) const {
  MS_LOG(WARNING) << "Create group in stub";
  auto executor = parallel::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  return executor->CreateCommGroup(group_name, ranks);
}

bool GroupManager::DestroyGroupByExecutor(const std::string &device_name, const std::string &group_name,
                                          uint32_t device_id) const {
  MS_LOG(WARNING) << "Destroy group in stub";
  auto executor = parallel::ExecutorManager::Instance().GetExecutor(device_name, device_id);
  MS_EXCEPTION_IF_NULL(executor);
  return executor->DestroyCommGroup(group_name);
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

bool hasCreated(const std::string &group_name, std::vector<uint32_t> ranks) {
  bool group_reused = false;
  const auto &pair = ParallelCommManager::GetInstance()->HcclGroups(ranks);
  if (pair.has_value()) {
    std::string new_group_name = pair.value().first;
    bool isCreated = pair.value().second;
    if (group_name == new_group_name && isCreated) {
      group_reused = true;
    }
  }
  return group_reused;
}

Status GroupManager::CreateGroup(const std::string &group_name, const std::vector<Device> &devices,
                                 mindspore::parallel::Group *const group) {
  // it is simple to use size to determine whether it is a world group
  if (devices.size() == g_device_manager->DeviceNum()) {
    auto iter = groups_.find(world_group_);
    if (iter == groups_.end()) {
      (void)group->Init(world_group_, devices);
      groups_[world_group_] = *group;
    } else {
      *group = iter->second;
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

    std::vector<uint32_t> ranks;
    (void)std::transform(std::begin(devices), std::end(devices), std::back_inserter(ranks),
                         [](const Device dev) { return static_cast<uint32_t>(dev.rank()); });
    // Create group through the executor
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

    std::pair<std::string, std::vector<uint32_t>> group_info = std::make_pair(group_name, ranks);
    group_info_.push_back(group_info);

    if (hasCreated(group_name, ranks)) {
      // If the group has already been created in Python, it will not be created again.
      std::string ranks_str = ParallelCommManager::GetInstance()->RankListName(ranks);
      MS_LOG(WARNING) << "The group '" << group_name << "' has been created, the ranks are: " << ranks_str;
      return Status::SUCCESS;
    }

    bool ret = CreateGroupByExecutor(device_name, group_name, ranks, device_id);
    if (!ret) {
      MS_LOG(WARNING) << "Create group failed, group name is " << group_name;
      return Status::FAILED;
    }

    MS_LOG(INFO) << "Create group success, group name is " << group_name;
    return Status::SUCCESS;
  }
}

Status GroupManager::DestroyGroup(const std::string &group_name) const {
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
  const std::string name = (*group).name();
  auto it = groups_.find(name);
  if (it == groups_.cend()) {
    MS_LOG(ERROR) << "Could not find group name :" << name;
    return Status::FAILED;
  }
  (void)groups_.erase(it);
  return DestroyGroup(name);
}

Status GroupManager::DestroyAllGroups() {
  for (auto it = groups_.cbegin(); it != groups_.cend(); ++it) {
    std::string name = it->first;
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
  if (it == groups_.cend()) {
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
  if (it == groups_.cend()) {
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
  if (it == groups_.cend()) {
    return Status::FAILED;
  }
  *group = &it->second;
  return Status::SUCCESS;
}

void GroupManager::Clear() { (void)DestroyAllGroups(); }
}  // namespace parallel
}  // namespace mindspore
