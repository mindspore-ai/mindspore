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

#include "parallel/device_manager.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
DeviceManagerPtr g_device_manager = nullptr;

Stage::Stage(const std::list<mindspore::parallel::Device>& devices, int num, int rank)
    : devices_(devices), number_(num), rank_(rank) {
  gm_ = GroupManager();
}

// NOTE: '-1' indicates ERROR
int Stage::global_rank(Group* g) const { return ((g == nullptr) ? rank_ : -1); }

bool InitDevice(int32_t device_num, int32_t global_rank, const std::string& backend) {
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
  if ((IntToUint(device_num) & IntToUint(device_num - 1)) != 0) {
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

  RankList devices, stage_map;
  for (int i = 0; i < device_num; ++i) {
    devices.push_back(i);
  }

  stage_map.push_back(device_num);
  g_device_manager = std::make_shared<DeviceManager>();
  if (g_device_manager->Init(devices, global_rank, stage_map, backend) == SUCCESS) {
    MS_LOG(INFO) << "Device initialization succeeds.";
    return true;
  } else {
    MS_LOG(ERROR) << "Device initialization fails.";
    return false;
  }
}

void CheckGlobalDeviceManager() {
  if (g_device_manager == nullptr) {
    MS_LOG(EXCEPTION) << "Device information has not been set!";
  }
}

int32_t GetListMemberByIndex(size_t index, const RankList& devices) {
  size_t i = 0;
  int32_t result = 0;
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

std::shared_ptr<Device> GetListMemberByIndex(size_t index, const std::list<std::shared_ptr<Device>>& device_list) {
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

// E.g. devices = [4, 5, 2, 1, 7, 8, 10], stage_map = [4, 3],
// therefore the stage_devices_ = [[4, 5, 2, 1], [7, 8, 10]].
Status DeviceManager::Init(const RankList& devices, int32_t global_device_rank, const RankList& stage_map,
                           const std::string& backend) {
  auto dev_it = devices.begin();
  auto stage_it = stage_map.begin();
  int32_t sum = 0;

  if ((backend != HCCL_BACKEND) && (backend != NCCL_BACKEND) && (backend != UNDEFINED_BACKEND)) {
    MS_LOG(ERROR) << "Invalid backend: " << backend;
    return Status::FAILED;
  }

  for (; stage_it != stage_map.end(); ++stage_it) {
    sum += (*stage_it);
  }
  if (IntToSize(sum) != devices.size()) {
    MS_LOG(ERROR) << "The number of 'devices' in the list is not equal to the mentioned "
                  << "size of 'stage_map'";
    return Status::FAILED;
  }

  for (; dev_it != devices.end(); ++dev_it) {
    std::shared_ptr<Device> one = std::make_shared<Device>(*dev_it);
    devices_.push_back(one);
  }

  size_t global_index = 0;
  for (stage_it = stage_map.begin(); stage_it != stage_map.end(); ++stage_it) {
    int num_device = *stage_it;
    if (num_device > MAX_DEVICE_NUM) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must not be greater than " << MAX_DEVICE_NUM;
      return Status::FAILED;
    }
    if (num_device <= 0) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must be positive";
      return Status::FAILED;
    }
    RankList curr_dev_list;
    for (int i = 0; i < num_device; ++i) {
      curr_dev_list.push_back(GetListMemberByIndex(global_index, devices));
      global_index++;
    }
    stage_devices_.push_back(curr_dev_list);
  }

  global_index = 0;
  for (stage_it = stage_map.begin(); stage_it != stage_map.end(); ++stage_it) {
    int num_device = *stage_it;
    if (num_device > MAX_DEVICE_NUM) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must be less than " << MAX_DEVICE_NUM;
      return Status::FAILED;
    }
    if (num_device <= 0) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must be positive";
      return Status::FAILED;
    }
    std::list<Device> curr_dev_list;
    for (int i = 0; i < num_device; ++i) {
      curr_dev_list.push_back(*GetListMemberByIndex(global_index, devices_));
      global_index++;
    }
    std::shared_ptr<Stage> new_stage = std::make_shared<Stage>(curr_dev_list);
    stages_.push_back(new_stage);
  }

  std::shared_ptr<Device> dev = std::make_shared<Device>(global_device_rank);
  device_ = dev;
  set_global_rank(global_device_rank);
  backend_ = backend;

  if (backend == HCCL_BACKEND) {
    gm_.set_world_group(HCCL_WORLD_GROUP);
  } else if (backend_ == NCCL_BACKEND) {
    gm_.set_world_group(NCCL_WORLD_GROUP);
  } else {
    gm_.set_world_group(UNDEFINED_WORLD_GROUP);
  }
  MS_LOG(INFO) << "The device num: " << devices.size() << "rank id: " << global_device_rank
               << "the backend: " << backend;
  return Status::SUCCESS;
}

std::shared_ptr<Stage> DeviceManager::GetStageById(int32_t stage_id) {
  std::shared_ptr<Stage> res;
  if (IntToSize(stage_id) >= stages_.size()) {
    MS_LOG(ERROR) << "the 'stage_id': " << stage_id << ", is out of the scope of 'stage_devices_': " << stages_.size();
    return res;
  }
  int32_t index = 0;
  for (auto& stage : stages_) {
    if (index == stage_id) return stage;
    index++;
  }
  return res;
}

RankList DeviceManager::GetDeviceListByStageId(int32_t stage_id) const {
  if (IntToSize(stage_id) >= stage_devices_.size())
    MS_LOG(ERROR) << "the 'stage_id': " << stage_id
                  << ", is out of the scope of 'stage_devices_': " << stage_devices_.size();
  RankList res;
  int32_t index = 0;
  for (auto& stage : stage_devices_) {
    if (index == stage_id) {
      return stage;
    }
    index++;
  }
  return res;
}

RankList DeviceManager::global_device_list(int32_t stage_id, int32_t rank, int32_t split_num) const {
  RankList res;
  if (split_num <= 0) {
    return res;
  }
  if (IntToSize(stage_id) >= stage_devices_.size()) {
    MS_LOG(ERROR) << "the 'stage_id': " << stage_id
                  << ", is out of the scope of 'stage_devices_': " << stage_devices_.size();
    return res;
  }

  RankList global_list = GetDeviceListByStageId(stage_id);
  if (global_list.size() % IntToSize(split_num)) {
    MS_LOG(ERROR) << "dev list size(" << global_list.size() << ") can not be divisible by split num: " << stage_id;
    return res;
  }

  std::vector<int32_t> dev_list;
  (void)std::copy(global_list.begin(), global_list.end(), std::back_inserter(dev_list));

  size_t index = 0;
  size_t slice_size = dev_list.size() / IntToSize(split_num);
  for (int32_t i = 0; i < split_num; ++i) {
    bool found = false;
    index = slice_size * IntToSize(i);
    for (size_t j = 0; j < slice_size; ++j) {
      if (dev_list[index + j] == rank) {
        found = true;
        break;
      }
    }

    if (found) {
      break;
    }
  }

  for (size_t k = 0; k < slice_size; ++k) {
    res.push_back(dev_list[index + k]);
  }
  return res;
}

Device DeviceManager::CreateNewDeviceByRank(int32_t rank) const { return Device(rank); }

std::list<Device> DeviceManager::CreateDeviceListByRankList(RankList ranks) {
  std::list<Device> dev_list;
  for (auto& rank : ranks) {
    Device one = CreateNewDeviceByRank(rank);
    dev_list.push_back(one);
  }
  return dev_list;
}

DeviceManager& DeviceManager::GetInstance() {
  static DeviceManager instance = DeviceManager();
  return instance;
}

std::string DeviceManager::FindRankListNameByHashName(const std::string& hash_name) {
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

std::string HashName(const std::string& origin_name) { return std::to_string(std::hash<string>{}(origin_name)); }

// Group name is generated using the increasing ranks of the devices.
// E.g. the devices' ranks are '<0, 5, 3, 7, 1>', and the generated group name
// is '0-1-3-5-7'.
std::string DeviceManager::GenerateGroupNameByRanks(RankList ranks) {
  std::string rank_list_name;
  std::list<int32_t>::iterator it;
  ranks.sort();  // sorted in increasing order
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
Group DeviceManager::CreateGroup(const std::string& group_name, const std::list<mindspore::parallel::Device>& devices) {
  if ((world_group() == NCCL_WORLD_GROUP) && (devices.size() != devices_.size())) {
    MS_LOG(EXCEPTION) << "Do not support sub group for nccl";
  }
  Group g;
  (void)gm_.CreateGroup(group_name, devices, &g);
  return g;
}

// Create the group with only the given devices' ranks.
Group DeviceManager::CreateGroup(const RankList& dev_ranks) {
  std::unordered_set<int32_t> rank_set(dev_ranks.begin(), dev_ranks.end());
  if (dev_ranks.size() != rank_set.size()) {
    MS_LOG(EXCEPTION) << "Invalid dev ranks(" << dev_ranks << "), it has the Duplicate elements in list";
  }

  std::string group_name = GenerateGroupNameByRanks(dev_ranks);
  std::list<Device> dev_list = CreateDeviceListByRankList(dev_ranks);
  return CreateGroup(group_name, dev_list);
}

void DeviceManager::Clear() {
  devices_.clear();
  stage_devices_.clear();
  gm_.Clear();
}

}  // namespace parallel
}  // namespace mindspore
