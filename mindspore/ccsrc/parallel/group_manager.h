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

#ifndef MINDSPORE_CCSRC_PARALLEL_GROUP_MANAGER_H_
#define MINDSPORE_CCSRC_PARALLEL_GROUP_MANAGER_H_

#include <cstdint>
#include <list>
#include <map>
#include <string>

#include "parallel/device.h"
#include "parallel/status.h"

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
  Status Init(const std::string& name, const std::list<Device>& devices);
  std::list<Device> GetDevicesList() const;
  std::string name() const { return name_; }
  bool IsInThisGroup(int32_t device_rank);
  Status GetIndex(size_t* index);
  size_t GetDevNum() const { return devices_.size(); }

 private:
  std::string name_;
  std::list<Device> devices_;
};

class GroupManager {
 public:
  GroupManager();
  ~GroupManager() = default;

  Status CreateGroup(const std::string& name, const std::list<Device>& devices, Group* group);
  Status DestroyGroup(Group* group);
  Status DestroyAllGroups();
  Status GetRankID(const std::string& name, unsigned int* rank_id);
  Status GetRankSize(const std::string& name, unsigned int* rank_size);
  Status FindGroup(const std::string& name, Group** group);
  std::string world_group() const { return world_group_; }
  void set_world_group(const std::string& name) { world_group_ = name; }
  void Clear();

 private:
  // the key is group name (name_)
  std::map<std::string, Group> groups_;
  std::string world_group_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_GROUP_MANAGER_H_
