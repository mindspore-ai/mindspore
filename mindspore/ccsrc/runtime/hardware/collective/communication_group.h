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
#include <memory>
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/utils/convert_utils_base.h"

namespace mindspore {
namespace device {
// The communication group for collecive operations. All of the collective communication happens within one specified
// communication group. MindSpore uses 'hccl_world_group' or 'nccl_world_group' as the default group.
class CommunicationGroup {
 public:
  explicit CommunicationGroup(const std::string name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank);
  virtual ~CommunicationGroup() {
    group_ranks_.clear();
    global_to_group_ranks_.clear();
    group_to_global_ranks_.clear();
  }

  // Initialize the communication group. For example, assign some hardware resources, etc.
  virtual bool Initialize(void *root_info) = 0;

  // Finalize the communication group. For example, destroy the group, etc.
  virtual bool Finalize() = 0;

  // Return the root rank's information. Only root rank of one group could call this method.Normally this is used for
  // collective libraries on the device side. For NCCL group, it returns 'ncclUniqueId'. For HCCL group, it returns
  // 'HcclRootInfo'.
  virtual void *GenerateRootInfo() { return nullptr; }

  // Get group or global rank for the given rank.
  uint32_t GetGroupRank(uint32_t global_rank);
  uint32_t GetGlobalRank(uint32_t group_rank);

  // Return the size of this communication group.
  uint32_t group_size() const;

 protected:
  // The third party collective communication libraries. They are dynamically loaded by MindSpore.
  const void *collective_comm_lib_ptr_;

  // Whether this communication group is initialized.
  bool initialized_;

  // This process's global rank.
  uint32_t global_rank_;

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
using CommunicationGroupPtr = std::shared_ptr<CommunicationGroup>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
