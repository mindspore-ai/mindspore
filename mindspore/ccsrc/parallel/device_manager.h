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

#ifndef MINDSPORE_CCSRC_PARALLEL_DEVICE_MANAGER_H_
#define MINDSPORE_CCSRC_PARALLEL_DEVICE_MANAGER_H_

#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "utils/convert_utils.h"
#include "common/utils.h"
#include "parallel/device.h"
#include "parallel/status.h"
#include "parallel/group_manager.h"
#include "parallel/strategy.h"
#include "parallel/device_matrix.h"

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

class Stage {
  // This class is used in pipeline-parallelization. Available devices are partitioned into multiple stages.
  // Currently, the function of pipeline-parallelization and this class are NOT implemented.
 public:
  explicit Stage(std::vector<Device> devices) : devices_(std::move(devices)), number_(0), rank_(0) {
    gm_ = GroupManager();
  }
  Stage(const std::vector<mindspore::parallel::Device>& devices, int num, int rank);
  ~Stage() = default;

  int GetStageNum() const { return number_; }
  size_t GetDevicesNum() const { return devices_.size(); }
  std::vector<Device> GetDevicesList() { return devices_; }
  int global_rank(Group* g) const;

 private:
  std::vector<Device> devices_;
  int number_;
  int32_t rank_;
  GroupManager gm_;
};

// This method is used for initializing the global DeviceManager 'g_device_manager',
// arguments including 'device_num' and 'global_rank'
bool InitDevice(int32_t device_num, int32_t global_rank, const std::string& backend);

void CheckGlobalDeviceManager();

std::string HashName(const std::string& rank_list_name);

class DeviceManager {
  // This class is used to manage the abstract devices, including group-related and stage-related management.
 public:
  DeviceManager() : local_rank_(0), global_rank_(0), stage_num_(0) { gm_ = GroupManager(); }
  ~DeviceManager() = default;

  Status Init(const RankList& devices, int32_t local_device, const RankList& stage_map, const std::string& backend);

  static DeviceManager& GetInstance();
  RankList GetDeviceListByStageId(int32_t stage_id) const;
  RankList global_device_list(int32_t stage_id, int32_t rank, int32_t split_num) const;

  Device CreateNewDeviceByRank(int32_t rank) const;
  std::vector<Device> CreateDeviceListByRankList(RankList ranks);

  std::string GenerateGroupNameByRanks(RankList dev_ranks);
  Group CreateGroup(const std::string& group_name, const std::vector<Device>& devices);
  Group CreateGroup(const RankList& dev_ranks);
  std::shared_ptr<Stage> GetStageById(int32_t stage_id);

  size_t DeviceNum() const { return devices_.size(); }

  int32_t GetStageNum() const { return static_cast<const int32_t>(stage_devices_.size()); }

  int32_t global_rank() const { return global_rank_; }
  std::string backend() const { return backend_; }
  void set_global_rank(int32_t global_rank) { global_rank_ = global_rank; }
  void Clear();
  std::string world_group() const { return gm_.world_group(); }
  std::string FindRankListNameByHashName(const std::string& hash_name);

 private:
  std::vector<std::shared_ptr<Device>> devices_;
  // each stage has a list of devices
  std::vector<std::vector<int32_t>> stage_devices_;
  std::shared_ptr<Device> device_;
  std::vector<std::shared_ptr<Stage>> stages_;
  GroupManager gm_;
  std::string backend_;

  // bimap:
  std::map<std::string, std::string> rank_to_group_;  // the key is rank list, value is hash name
  std::map<std::string, std::string> group_to_rank_;  // the key is hash name, value is rank list

  int32_t local_rank_;
  int32_t global_rank_;
  int32_t stage_num_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_DEVICE_MANAGER_H_
