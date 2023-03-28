/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <vector>
#include <unordered_map>

#include "utils/hash_set.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
DeviceManagerPtr g_device_manager = nullptr;

bool CheckDeviceConfig(int64_t device_num, int64_t global_rank, const std::string &backend,
                       const std::vector<int64_t> &stage) {
  if (device_num <= 0) {
    MS_LOG(ERROR) << "The context configuration parameter 'device_num' must be positive, "
                     "but got the value of device_num: "
                  << device_num;
    return false;
  }
  if (global_rank < 0) {
    MS_LOG(ERROR) << "The context configuration parameter 'global_rank' must be nonnegative, "
                     "but got the value of global_rank: "
                  << global_rank;
    return false;
  }
  if (device_num > MAX_DEVICE_NUM) {
    MS_LOG(ERROR) << "The context configuration parameter 'device_num' must be no more than " << MAX_DEVICE_NUM
                  << ", but got the value of device_num: " << device_num;
    return false;
  }
  // 'device_num_converted' must be divisible by 8
  if (LongToSize(device_num) % DEVICE_NUM_PER_SERVER != 0 && device_num != 1 && device_num != 2 && device_num != 4) {
    MS_LOG(ERROR) << "The context configuration parameter device_num' must be divisible by 8, "
                     "or equal to 1, 2 or 4, but got the value of device_num: "
                  << device_num;
    return false;
  }
  if (global_rank >= device_num) {
    MS_LOG(ERROR) << "The context configuration parameter 'global_rank' must be less than 'device_num', "
                     "but got the value of global_rank: "
                  << global_rank << ", and the value of device_num: " << device_num;
    return false;
  }
  if ((backend != HCCL_BACKEND) && (backend != NCCL_BACKEND) && (backend != UNDEFINED_BACKEND)) {
    MS_LOG(ERROR) << "For 'InitDevice', the argument 'backend' must be hccl, nccl "
                     "or undefined_backend, but got invalid backend: "
                  << backend;
    return false;
  }
  if (stage.empty()) {
    MS_LOG(ERROR) << "The size of parameter 'stage' must be positive, but got the size of stage is empty.";
    return false;
  }
  return true;
}

bool InitDevice(int64_t device_num, int64_t global_rank, const std::string &backend,
                const std::vector<int64_t> &stage) {
  if (!CheckDeviceConfig(device_num, global_rank, backend, stage)) {
    return false;
  }

  RankList devices, stage_map;
  for (int64_t i = 0; i < device_num; ++i) {
    devices.push_back(i);
  }

  int64_t summed_value = 0;
  for (auto begin = stage.begin(); begin != stage.end(); ++begin) {
    if (*begin <= 0) {
      MS_LOG(ERROR) << "The value in the pipeline stages should be positive value, but got the value: " << *begin;
      return false;
    }
    summed_value += *begin;
    stage_map.push_back(*begin);
  }

  if (summed_value != device_num) {
    MS_LOG(ERROR) << "The sum of the pipeline stage must be equal to the device_num, "
                     "but got sum of the pipeline stage :"
                  << summed_value << " and the device_num : " << device_num;
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

namespace {
constexpr int64_t NODE_PER_SERVER = 8;
Status IsFeasibleDeiveListOneServer(const RankList &rank_list) {
  if (rank_list.size() == 1 || rank_list.size() == 8) {
    return SUCCESS;
  }
  if (rank_list.size() == 4 && (rank_list[3] - rank_list[0] == 3) && (rank_list[0] == 0 || rank_list[3] == 7)) {
    return SUCCESS;
  }
  if (rank_list.size() == 4 && (rank_list[3] % 4 == rank_list[1] % 4) && (rank_list[2] % 4 == rank_list[0] % 4)) {
    return SUCCESS;
  }
  if (rank_list.size() == 2) {
    if (rank_list[1] - rank_list[0] == 4) {
      return SUCCESS;
    }
    if (rank_list[1] < 4 && rank_list[0] < 4) {
      return SUCCESS;
    }
    if (rank_list[1] >= 4 && rank_list[0] >= 4) {
      return SUCCESS;
    }
  }
  return FAILED;
}

Status IsFeasibleDeiveList(const RankList &rank_list) {
  std::unordered_map<int64_t, RankList> server_ranks_map;
  for (auto rank : rank_list) {
    int64_t server_id = rank / NODE_PER_SERVER;
    int64_t local_rank = rank % NODE_PER_SERVER;
    server_ranks_map[server_id].push_back(local_rank);
  }
  std::vector<RankList> server_ranks_list;
  (void)std::transform(server_ranks_map.begin(), server_ranks_map.end(), std::back_inserter(server_ranks_list),
                       [](auto pairs) { return pairs.second; });
  auto server0_local_ranks = server_ranks_list[0];
  bool is_all_server_same_count =
    std::all_of(server_ranks_list.begin(), server_ranks_list.end(),
                [&server0_local_ranks](auto ranks) { return ranks == server0_local_ranks; });
  if (!is_all_server_same_count) {
    MS_LOG(INFO) << "All server should has the same ranks, which means rank_id % 8 in each server should be the same. "
                    "current rank list is"
                 << rank_list;
    return FAILED;
  }
  return IsFeasibleDeiveListOneServer(server0_local_ranks);
}
}  // namespace

Status DeviceManager::CheckDeviceList(const RankList &rank_list) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice || backend == kDavinciDevice) {
    return IsFeasibleDeiveList(rank_list);
  }
  return SUCCESS;
}

// E.g. devices = [0, 1, 2, 3, 4, 5, 6, 7], stage_map = [4, 4],
// therefore the stage_devices_ = [[0, 1, 2, 3], [4, 5, 6, 7]].
Status DeviceManager::Init(const RankList &devices, int64_t global_device_rank, const RankList &stage_map,
                           const std::string &backend) {
  if ((backend != HCCL_BACKEND) && (backend != NCCL_BACKEND) && (backend != UNDEFINED_BACKEND)) {
    MS_LOG(ERROR) << "For 'Init', the argument 'backend' must be hccl, nccl "
                     "or undefined_backend, but got invalid backend: "
                  << backend;
    return FAILED;
  }

  if (stage_map.empty() || devices.empty()) {
    MS_LOG(ERROR) << "The size of stage_map and devices must be positive, but got the size of stage_map: "
                  << stage_map.size() << ", and the size of devices : " << devices.size();
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
      MS_LOG(ERROR) << "The number of 'devices' in a stage must not be greater than " << MAX_DEVICE_NUM
                    << ", but got the number of 'devices' in a stage: " << num_device;
      return FAILED;
    }
    if (num_device <= 0) {
      MS_LOG(ERROR) << "The number of 'devices' in a stage must be positive, but got the num_device: " << num_device;
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

RankList DeviceManager::GetDeviceListBetweenStage() const {
  std::vector<int64_t> rank_list;
  auto rank_id = g_device_manager->global_rank();
  auto stage_id = g_device_manager->stage_id();
  auto stage_num = g_device_manager->stage_num();
  if (stage_num < 1) {
    MS_LOG(EXCEPTION) << "Stage num got " << stage_num << ", expected a positive integer.";
  }
  auto device_num = DeviceNum();
  auto per_stage_rank_num = device_num / LongToSize(stage_num);
  for (int64_t i = 0; i < stage_num; ++i) {
    rank_list.push_back(rank_id + SizeToLong(per_stage_rank_num) * (i - stage_id));
  }
  return rank_list;
}

RankList DeviceManager::GetDeviceListByStageId(int64_t stage_id) const {
  if (LongToSize(stage_id) >= stage_devices_.size()) {
    MS_LOG(ERROR) << "the 'stage_id': " << stage_id
                  << ", is out of the scope of 'stage_devices_': " << stage_devices_.size();
  }
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

std::vector<Device> DeviceManager::CreateDeviceListByRankList(RankList ranks) const {
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
  std::map<std::string, std::string>::const_iterator iter = group_to_rank_.find(hash_name);
  if (iter == group_to_rank_.cend()) {
    MS_LOG(WARNING) << "Can not find the rank list name by hash name: " << hash_name;
    return tmp;
  }
  return iter->second;
}

RankList DeviceManager::FindRankListByHashName(const std::string &hash_name) {
  std::string rank_list_name = FindRankListNameByHashName(hash_name);
  if (rank_list_name == "WORLD_GROUP") {
    int64_t device_num = SizeToLong(g_device_manager->DeviceNum());
    RankList rank_list;
    for (size_t i = 0; i < size_t(device_num); ++i) {
      rank_list.push_back(i);
    }
    return rank_list;
  }
  RankList rank_list;
  std::string rank_str = "";
  rank_list_name = rank_list_name + "-";
  for (size_t i = 0; i < rank_list_name.size(); i++) {
    if (rank_list_name[i] == '-') {
      int64_t rank_id = std::stoi(rank_str);
      rank_list.push_back(rank_id);
      rank_str = "";
    } else if (rank_list_name[i] <= '9' && rank_list_name[i] >= '0') {
      rank_str.push_back(rank_list_name[i]);
    } else {
      MS_LOG(EXCEPTION) << "The rank list name cannot convert to rank list: " << rank_list_name;
    }
  }
  return rank_list;
}

std::string HashName(const std::string &origin_name) { return std::to_string(std::hash<string>{}(origin_name)); }

std::string RankListName(const RankList &ranks) {
  std::string rank_list_name;
  for (auto it = ranks.begin(); it != ranks.end(); ++it) {
    if (it == ranks.begin()) {
      rank_list_name = std::to_string(*it);
    } else {
      rank_list_name += "-" + std::to_string(*it);
    }
  }
  return rank_list_name;
}

// Group name is generated using the increasing ranks of the devices.
// E.g. the devices' ranks are '<0, 5, 3, 7, 1>', and the generated group name
// is '0-1-3-5-7'.
std::string DeviceManager::GenerateGroupNameByRanks(RankList ranks) {
  std::sort(ranks.begin(), ranks.end());  // sorted in increasing order
  std::string rank_list_name = RankListName(ranks);

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
Status DeviceManager::CreateGroup(const std::string &group_name,
                                  const std::vector<mindspore::parallel::Device> &devices, Group *const comm_group) {
  RankList rank_list;
  (void)std::transform(devices.begin(), devices.end(), std::back_inserter(rank_list),
                       [](const Device &device) { return device.rank(); });
  if (CheckDeviceList(rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "Create communication group failed, the rank list is: " << rank_list;
    return FAILED;
  }
  if (gm_.CreateGroup(group_name, devices, comm_group) != SUCCESS) {
    return FAILED;
  }
  group_to_rank_[group_name] = RankListName(rank_list);
  return SUCCESS;
}

// Create the group with only the given devices' ranks.
Status DeviceManager::CreateGroup(const RankList &dev_ranks, Group *const comm_group) {
  mindspore::HashSet<int64_t> rank_set(dev_ranks.begin(), dev_ranks.end());
  if (dev_ranks.size() != rank_set.size()) {
    MS_LOG(ERROR) << "Invalid dev ranks(" << dev_ranks << "), it has the Duplicate elements in list";
    return FAILED;
  }
  if (CheckDeviceList(dev_ranks) != SUCCESS) {
    MS_LOG(ERROR) << "Create communication group failed, the rank list is: " << dev_ranks;
    return FAILED;
  }
  std::string group_name = GenerateGroupNameByRanks(dev_ranks);
  auto dev_list = CreateDeviceListByRankList(dev_ranks);
  return CreateGroup(group_name, dev_list, comm_group);
}

void DeviceManager::Clear() {
  devices_.clear();
  stage_devices_.clear();
  gm_.Clear();
}
}  // namespace parallel
}  // namespace mindspore
