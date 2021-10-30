/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_

#include <map>
#include <string>
#include <vector>
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore {
namespace device {
// The communication group for collecive operations. All of the collective communication happens within one specified
// communication group. MindSpore uses 'hccl_world_group' or 'nccl_world_group' as the default group.
class CommunicationGroup {
 public:
  explicit CommunicationGroup(uint32_t size, const std::string name, const std::vector<uint32_t> &group_ranks)
      : size_(size), name_(name), group_ranks_(group_ranks), global_to_group_ranks_({}), group_to_global_ranks_({}) {}

  virtual ~CommunicationGroup() {
    group_ranks_.clear();
    global_to_group_ranks_.clear();
    group_to_global_ranks_.clear();
  }

  // Initialize the communication group. For example, assign some hardware resources, etc.
  virtual void Initialize() = 0;

  // Finalize the communication group. For example, destroy the group, etc.
  virtual void Finalize() = 0;

  // Get group or global rank for the given rank.
  uint32_t GetGroupRank(uint32_t global_rank);
  uint32_t GetGlobalRank(uint32_t group_rank);

  // Return the size of this communication group.
  uint32_t group_size() const;

 protected:
  // The number of processes in this communication group.
  uint32_t size_;

  // This group's name.
  std::string name_;

  // The global rank list of the processes in this group.
  std::vector<uint32_t> group_ranks_;

  // The mapping of global ranks and group ranks.
  std::map<uint32_t, uint32_t> global_to_group_ranks_;
  std::map<uint32_t, uint32_t> group_to_global_ranks_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
