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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MANAGER_H_

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/strategy.h"
#include "utils/convert_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace parallel {
#define MAX_DEVICE_NUM 1024

constexpr char HCCL_BACKEND[] = "hccl";
constexpr char NCCL_BACKEND[] = "nccl";
constexpr char UNDEFINED_BACKEND[] = "undefined_backend";

class DeviceManager;
using DeviceManagerPtr = std::shared_ptr<DeviceManager>;
// 'g_device_manager' is the globally unique manager to manage the devices.
extern DeviceManagerPtr g_device_manager;

// This method is used for initializing the global DeviceManager 'g_device_manager',
// arguments including 'device_num' and 'global_rank'
bool InitDevice(int64_t device_num, int64_t global_rank, const std::string &backend, const std::vector<int64_t> &stage);

void CheckGlobalDeviceManager();

std::string HashName(const std::string &rank_list_name);

class DeviceManager {
  // This class is used to manage the abstract devices, including group-related and stage-related management.
 public:
  DeviceManager() { gm_ = GroupManager(); }
  ~DeviceManager() = default;

  Status Init(const RankList &devices, int64_t local_device, const RankList &stage_map, const std::string &backend);

  static DeviceManager &GetInstance();
  RankList GetDeviceListByStageId(int64_t stage_id) const;
  RankList GetDeviceListInThisStage() const;

  Device CreateNewDeviceByRank(int64_t rank) const;
  std::vector<Device> CreateDeviceListByRankList(RankList ranks);

  std::string GenerateGroupNameByRanks(RankList dev_ranks);
  Group CreateGroup(const std::string &group_name, const std::vector<Device> &devices);
  Group CreateGroup(const RankList &dev_ranks);

  size_t DeviceNum() const { return devices_.size(); }
  int64_t stage_num() const { return stage_num_; }
  int64_t stage_device_num() const { return stage_device_num_; }
  int64_t stage_id() const { return stage_id_; }
  int64_t rank_index_in_stage() const { return rank_index_in_stage_; }
  int64_t global_rank() const { return global_rank_; }
  std::string backend() const { return backend_; }

  void Clear();
  std::string world_group() const { return gm_.world_group(); }
  std::vector<std::pair<std::string, std::vector<uint32_t>>> group_info() const { return gm_.group_info(); }
  std::string FindRankListNameByHashName(const std::string &hash_name);

 private:
  std::vector<std::shared_ptr<Device>> devices_;
  // each stage has a list of devices
  std::vector<std::vector<int64_t>> stage_devices_;
  std::shared_ptr<Device> device_;
  GroupManager gm_;
  std::string backend_;

  // bimap:
  std::map<std::string, std::string> rank_to_group_;  // the key is rank list, value is hash name
  std::map<std::string, std::string> group_to_rank_;  // the key is hash name, value is rank list

  int64_t global_rank_ = 0;          // the real rank in all devices
  int64_t stage_num_ = 1;            // the stage num
  int64_t stage_id_ = 0;             // the stage id of the global_rank_
  int64_t rank_index_in_stage_ = 0;  // the index of this rank in it's stage
  int64_t stage_device_num_ = 0;     // the device num of one stage
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MANAGER_H_
