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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COLLECTIVE_COMMUNICATION_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COLLECTIVE_COMMUNICATION_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "runtime/hardware/collective/communication_group.h"

namespace mindspore {
namespace device {
// The base class of collective communication library.
// For collective communication on the device side like GPU, the entry is NvidiaCollectiveCommLib which calls NCCL.
// For collective communication on the host side, the entry is MPICollectiveCommLib which call OpenMPI, or
// MsCollectiveCommLib which uses the host-side communication library developed by MindSpore.
class CollectiveCommunicationLib {
 public:
  CollectiveCommunicationLib() : global_rank_id_(0), local_rank_id_(0), global_rank_size_(0) {}
  virtual ~CollectiveCommunicationLib() { groups_.clear(); }

  // Initialize collecitve communication library.
  // Input 'global_rank' represents this process's global rank.
  // Normally, collective communication libraries on host side will generate this rank inside the 'Initialize' method.
  // But collective communication libraries on device side needs this input passed by the caller.
  virtual void Initialize(uint32_t global_rank = UINT32_MAX) { return; }

  // Finalize collecitve communication library.
  virtual void Finalize() { return; }

  // Create communication group. This is the precondition for all collective operations on both host and device side.
  virtual bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) {
    return true;
  }

  // Destroy the communication group.
  virtual bool DestroyCommunicationGroup(const std::string &group_name);

  // Get the rank id of this process in the specified group.
  uint32_t GetRankId(const std::string &group_name);

  // Get the size of the specified group.
  uint32_t GetGroupSize(const std::string &group_name);

  // Returns the local rank id of this process.
  uint32_t local_rank_id() const;

 protected:
  // The global rank id of this process. Normally this range is 0 to `total process number - 1`.
  uint32_t global_rank_id_;

  // The local rank id of this process within the same node. This is usually used as device id.
  uint32_t local_rank_id_;

  // The global rank size. Normally this is equal to `total process number`.
  uint32_t global_rank_size_;

  // This map stores the groups which will be accessed and used by the caller.
  std::map<std::string, std::shared_ptr<CommunicationGroup>> groups_;
};
using CollectiveCommunicationLibPtr = CollectiveCommunicationLib *;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COLLECTIVE_COMMUNICATION_LIB_H_
