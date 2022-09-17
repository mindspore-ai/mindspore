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
#include <sstream>
#include <algorithm>
#include "utils/log_adapter.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace device {
// The communication group for collecive operations. All of the collective communication happens within one specified
// communication group. MindSpore uses 'hccl_world_group' or 'nccl_world_group' as the default group.
class CommunicationGroup {
 public:
  explicit CommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank,
                              uint32_t local_group_rank, uint32_t local_group_size);
  virtual ~CommunicationGroup() {
    group_ranks_.clear();
    global_to_group_ranks_.clear();
    group_to_global_ranks_.clear();
  }

  // Initialize the communication group. For example, assign some hardware resources, etc.
  virtual bool Initialize(void *root_info) = 0;

  // Finalize the communication group. For example, destroy the group, etc.
  virtual bool Finalize() = 0;

  // Return the root rank's information and its size. Normally this is used for collective libraries on the device side.
  // For NCCL group, it returns a pointer to 'ncclUniqueId'. For HCCL group, it returns a pointer to 'HcclRootInfo'.
  virtual void *GenerateRootInfo(size_t *root_info_size) { return nullptr; }

  // Get group or global rank for the given rank.
  uint32_t GetGroupRank(uint32_t global_rank);
  uint32_t GetGlobalRank(uint32_t group_rank);
  uint32_t GetLocalGroupRank();

  // Return the size of this communication group.
  uint32_t group_size() const;
  uint32_t local_group_size() const;
  virtual void set_local_rank(uint32_t local_group_rank) { local_group_rank_ = local_group_rank; }
  virtual void set_local_size(uint32_t local_group_size) { local_group_size_ = local_group_size; }

  // Return group ranks info.
  const std::vector<uint32_t> &group_ranks() const;
  const std::map<uint32_t, uint32_t> &global_to_group_ranks() const;
  const std::map<uint32_t, uint32_t> &group_to_global_ranks() const;

 protected:
  // Whether this communication group is initialized.
  bool initialized_;

  // This process's global rank.
  uint32_t global_rank_;

  // This group's local rank in current server.
  uint32_t local_group_rank_;

  // This group's local size in current server.
  uint32_t local_group_size_;

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

#define CHECK_RET(expression, result, message)                                                     \
  do {                                                                                             \
    auto ret = (expression);                                                                       \
    if (ret != (result)) {                                                                         \
      std::ostringstream oss;                                                                      \
      oss << "Error in file " << __FILE__ << " | Error on line " << __LINE__ << ": " << (message); \
      pybind11::pybind11_fail(oss.str());                                                          \
    }                                                                                              \
  } while (0)

#define CHECK_IF_NULL(ptr)                                                                               \
  do {                                                                                                   \
    if ((ptr) == nullptr) {                                                                              \
      std::ostringstream oss;                                                                            \
      oss << "Error in file " << __FILE__ << " | Error on line " << __LINE__ << ": The pointer[" << #ptr \
          << "] is null.";                                                                               \
      pybind11::pybind11_fail(oss.str());                                                                \
    }                                                                                                    \
  } while (0)
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
