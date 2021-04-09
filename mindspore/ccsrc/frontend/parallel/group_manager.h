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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GROUP_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GROUP_MANAGER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "frontend/parallel/device.h"
#include "frontend/parallel/status.h"

namespace mindspore {
namespace parallel {
constexpr char HCCL_WORLD_GROUP[] = "hccl_world_group";
constexpr char NCCL_WORLD_GROUP[] = "nccl_world_group";
constexpr char UNDEFINED_WORLD_GROUP[] = "undefined_world_group";

// Devices that need communication should in the same group. These classes are used to
// create and destroy group among devices.
class Group {
 public:
  Group();
  ~Group() = default;
  Status Init(const std::string &name, const std::vector<Device> &devices);
  std::vector<Device> GetDevicesList() const;
  std::string name() const { return name_; }
  bool IsInThisGroup(int64_t device_rank);
  Status GetIndex(size_t *index);
  size_t GetDevNum() const { return devices_.size(); }

 private:
  std::string name_;
  std::vector<Device> devices_;
};

class GroupManager {
 public:
  GroupManager();
  ~GroupManager() = default;

  Status CreateGroup(const std::string &name, const std::vector<Device> &devices, Group *group);
  Status DestroyGroup(Group *group);
  Status DestroyAllGroups();
  Status GetRankID(const std::string &name, uint32_t *rank_id);
  Status GetRankSize(const std::string &name, uint32_t *rank_size);
  Status FindGroup(const std::string &name, Group **group);
  std::string world_group() const { return world_group_; }
  void set_world_group(const std::string &name) { world_group_ = name; }
  std::vector<std::pair<std::string, std::vector<uint32_t>>> group_info() const { return group_info_; }
  void Clear();

 private:
  bool CreateGroupByExecutor(const std::string &device_name, const std::string &group_name,
                             const std::vector<uint32_t> ranks, int device_id);
  bool DestroyGroupByExecutor(const std::string &device_name, const std::string &group_name, int device_id);
  Status DestroyGroup(const std::string &group_name);
  // the key is group name (name_)
  std::map<std::string, Group> groups_;
  std::string world_group_;
  std::vector<std::pair<std::string, std::vector<uint32_t>>> group_info_;
};

Status CreateGroups(const std::vector<std::pair<std::string, std::vector<uint32_t>>> &group_info);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GROUP_MANAGER_H_
