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

#include "frontend/parallel/device_manager.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
DeviceManagerPtr g_device_manager = nullptr;
bool InitDevice(int64_t device_num, int64_t global_rank, const std::string &backend,
                const std::vector<int64_t> &stage) {
  if (device_num <= 0) {
    MS_LOG(ERROR) << "'device_num' must be positive.";
    return false;
  }
  if (global_rank < 0) {
    MS_LOG(ERROR) << "'global_rank' must be nonnegative.";
    return false;
  }
  if (device_num > MAX_DEVICE_NUM) {
    MS_LOG(ERROR) << "'device_num' must be no more than " << MAX_DEVICE_NUM << ".";
    return false;
  }
  // 'device_num_converted' must be the power of 2
  if ((LongToUlong(device_num) & LongToUlong(device_num - 1)) != 0) {
    MS_LOG(ERROR) << "'device_num' must be the power of 2.";
    return false;
  }
  if (global_rank >= device_num) {
    MS_LOG(ERROR) << "'global_rank' must be less than 'device_num'.";
    return false;
  }
  if ((backend != HCCL_BACKEND) && (backend != NCCL_BACKEND) && (backend != UNDEFINED_BACKEND)) {
    MS_LOG(ERROR) << "Invalid backend: " << backend;
    return false;
  }
  if (stage.empty()) {
    MS_LOG(ERROR) << "The size of stage must be positive";
    return false;
  }

  RankList devices, stage_map;
  for (int64_t i = 0; i < device_num; ++i) {
    devices.push_back(i);
  }

  int64_t summed_value = 0;
  for (auto begin = stage.begin(); begin != stage.end(); ++begin) {
    if (*begin <= 0) {
      MS_LOG(ERROR) << "The value in the pipeline stages should be positive value";
      return false;
    }
    summed_value += *begin;
    stage_map.push_back(*begin);
  }

  if (summed_value != device_num) {
    MS_LOG(ERROR) << "The sum of the pipeline stage :" << summed_value << " is not equal to the device_num "
                  << device_num;
    return false;
  }

  for (auto &ele : stage_map) {
    MS_LOG(DEBUG) << "Obtained stage id: " << ele;
  }
  if (g_device_manager) {
    auto gm = g_device_manager->group_manager();
    g_device_manager = std::make_shared<DeviceManager>();
    g_device_manager->set_group_manager(gm);
  } else {
    g_device_manager = std::make_shared<DeviceManager>();
  }
  if (g_device_manager->Init(devices, global_rank, stage_map, backend) == SUCCESS) {
    MS_LOG(INFO) << "Device initialization succeeds.";
    return true;
  }

  MS_LOG(ERROR) << "Device initialization fails.";
  return false;
}

void CheckGlobalDeviceManager() {
  if (g_device_manager == nullptr) {
    MS_LOG(EXCEPTION) << "Device information has not been set!";
  }
}

int64_t GetListMemberByIndex(size_t index, const RankList &devices) {
  size_t i = 0;
  int64_t result = 0;
  if ((devices.empty()) || (index >= devices.size())) {
    MS_LOG(EXCEPTION) << "Index is out of the list scope";
  }
  auto it = devices.begin();
  for (; it != devices.end(); ++it) {
    if (i == index) {
      result = *it;
      break;
    }
    ++i;
  }
  return result;
}

std::shared_ptr<Device> GetListMemberByIndex(size_t index, const std::vector<std::shared_ptr<Device>> &device_list) {
  size_t i = 0;
  std::shared_ptr<Device> result;
  if ((device_list.empty()) || (index >= device_list.size())) {
    MS_LOG(EXCEPTION) << "Index is out of the list scope";
  }
  auto it = device_list.begin();
  for (; it != device_list.end(); ++it) {
    if (i == index) {
      result = *it;
      break;
    }
    ++i;
  }
  return result;
}

// E.g. devices = [0, 1, 2, 3, 4, 5, 6, 7], stage_map = [4, 4],
// therefore the stage_devices_ = [[0, 1, 2, 3], [4, 5, 6, 7]].
Status DeviceManager::Init(const RankList &devices, int64_t global_device_rank, const RankList &stage_map,
                           const std::string &backend) {
  if ((backend != HCCL_BACKEND) && (backend != NCCL_BACKEND) && (backend != UNDEFINED_BACKEND)) {
    MS_LOG(ERROR) << "Invalid backend: " << backend;
    return FAILED;
  }

  if (stage_map.empty() || devices.empty()) {
    MS_LOG(ERROR) << "The size of stage_map and devices must be positive";
    return FAILED;
  }

  for (auto &dev : devices) {
    std::shared_ptr<Device> one = std::make_shared<Device>(dev);
    devices_.push_back(one);
  }

  size_t global_index = 0;
  for (auto &stage : stage_map) {
    int64_t num_device = stage;
    if (num_device > MAX_DEVICE_NUM) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must not be greater than " << MAX_DEVICE_NUM;
      return FAILED;
    }
    if (num_device <= 0) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must be positive";
      return FAILED;
    }
    RankList curr_dev_list;
    for (int64_t i = 0; i < num_device; ++i) {
      curr_dev_list.push_back(GetListMemberByIndex(global_index, devices));
      global_index++;
    }
    stage_devices_.push_back(curr_dev_list);
  }

  std::shared_ptr<Device> dev = std::make_shared<Device>(global_device_rank);
  device_ = dev;

  global_rank_ = global_device_rank;
  stage_num_ = static_cast<const int64_t>(stage_map.size());
  stage_id_ = global_device_rank / static_cast<const int64_t>(devices.size() / stage_map.size());
  rank_index_in_stage_ = global_rank_ - stage_id_ * (static_cast<const int64_t>(devices.size()) / stage_num_);
  stage_device_num_ = static_cast<const int64_t>(devices.size()) / stage_num_;

  backend_ = backend;

  if (backend == HCCL_BACKEND) {
    gm_.set_world_group(HCCL_WORLD_GROUP);
  } else if (backend_ == NCCL_BACKEND) {
    gm_.set_world_group(NCCL_WORLD_GROUP);
  } else {
    gm_.set_world_group(UNDEFINED_WORLD_GROUP);
  }
  MS_LOG(INFO) << "The device num: " << devices.size() << ", rank id: " << global_device_rank
               << ", the backend: " << backend << ", the stage num: " << stage_num_ << ", the stage id: " << stage_id_
               << ", the rank index in stage is: " << rank_index_in_stage_;
  return SUCCESS;
}

RankList DeviceManager::GetDeviceListInThisStage() const { return GetDeviceListByStageId(stage_id_); }

RankList DeviceManager::GetDeviceListByStageId(int64_t stage_id) const {
  if (LongToSize(stage_id) >= stage_devices_.size())
    MS_LOG(ERROR) << "the 'stage_id': " << stage_id
                  << ", is out of the scope of 'stage_devices_': " << stage_devices_.size();
  RankList res;
  int64_t index = 0;
  for (auto &stage : stage_devices_) {
    if (index == stage_id) {
      return stage;
    }
    index++;
  }
  return res;
}

Device DeviceManager::CreateNewDeviceByRank(int64_t rank) const { return Device(rank); }

std::vector<Device> DeviceManager::CreateDeviceListByRankList(RankList ranks) {
  std::vector<Device> dev_list;
  for (auto &rank : ranks) {
    Device one = CreateNewDeviceByRank(rank);
    dev_list.push_back(one);
  }
  return dev_list;
}

DeviceManager &DeviceManager::GetInstance() {
  static DeviceManager instance = DeviceManager();
  return instance;
}

std::string DeviceManager::FindRankListNameByHashName(const std::string &hash_name) {
  std::string tmp = "WORLD_GROUP";
  if ((hash_name == HCCL_WORLD_GROUP) || (hash_name == NCCL_WORLD_GROUP)) {
    return tmp;
  }
  auto iter = group_to_rank_.find(hash_name);
  if (iter == group_to_rank_.end()) {
    MS_LOG(WARNING) << "Can not find the rank list name by hash name: " << hash_name;
    return tmp;
  }
  return iter->second;
}

std::string HashName(const std::string &origin_name) { return std::to_string(std::hash<string>{}(origin_name)); }

// Group name is generated using the increasing ranks of the devices.
// E.g. the devices' ranks are '<0, 5, 3, 7, 1>', and the generated group name
// is '0-1-3-5-7'.
std::string DeviceManager::GenerateGroupNameByRanks(RankList ranks) {
  std::string rank_list_name;
  std::vector<int64_t>::iterator it;
  std::sort(ranks.begin(), ranks.end());  // sorted in increasing order
  for (it = ranks.begin(); it != ranks.end(); ++it) {
    if (it == ranks.begin()) {
      rank_list_name = std::to_string(*it);
    } else {
      rank_list_name += "-" + std::to_string(*it);
    }
  }

  // hash rank-list-name and add ranks' size as prefix
  std::string group_hash_name = HashName(rank_list_name);
  std::string group_name = std::to_string(ranks.size()) + "-" + group_hash_name;

  if (rank_to_group_.find(rank_list_name) == rank_to_group_.end()) {
    if (group_to_rank_.find(group_name) == group_to_rank_.end()) {
      rank_to_group_[rank_list_name] = group_name;
      group_to_rank_[group_name] = rank_list_name;
      MS_LOG(INFO) << "The rank list name is " << rank_list_name << "nd group name is " << group_name;
    } else {
      MS_LOG(EXCEPTION) << "Hash collision, the current rank list: " << rank_list_name
                        << "the old rank list:" << group_to_rank_.find(group_name)->second
                        << "the group name: " << group_name;
    }
  }
  return group_name;
}

// Create the group with the given devices and the given name. The GroupManager
// gm_ will create a new group only if there does not exit a group with the same
// name. Otherwise, let the pointer g point to that group.
Group DeviceManager::CreateGroup(const std::string &group_name,
                                 const std::vector<mindspore::parallel::Device> &devices) {
  Group g;
  (void)gm_.CreateGroup(group_name, devices, &g);
  return g;
}

// Create the group with only the given devices' ranks.
Group DeviceManager::CreateGroup(const RankList &dev_ranks) {
  std::unordered_set<int64_t> rank_set(dev_ranks.begin(), dev_ranks.end());
  if (dev_ranks.size() != rank_set.size()) {
    MS_LOG(EXCEPTION) << "Invalid dev ranks(" << dev_ranks << "), it has the Duplicate elements in list";
  }

  std::string group_name = GenerateGroupNameByRanks(dev_ranks);
  auto dev_list = CreateDeviceListByRankList(dev_ranks);
  return CreateGroup(group_name, dev_list);
}

void DeviceManager::Clear() {
  devices_.clear();
  stage_devices_.clear();
  gm_.Clear();
}
}  // namespace parallel
}  // namespace mindspore
