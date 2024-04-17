/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_DUMMY_COLLECTIVE_COMMUNICATION_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_DUMMY_COLLECTIVE_COMMUNICATION_LIB_H_

#include <string>
#include <vector>

#include "runtime/collective/communication_group.h"
#include "runtime/collective/collective_communication_lib.h"

namespace mindspore {
namespace device {
///
/// \brief DummyCommunicationGroup to maintain group device relationship.
///
class DummyCommunicationGroup : public CommunicationGroup {
 public:
  DummyCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank,
                          uint32_t local_group_rank, uint32_t local_group_size)
      : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size) {}

  ~DummyCommunicationGroup() override = default;

  bool Initialize(void *root_info) override {
    if (root_info == nullptr) {
      MS_LOG_WARNING << "Initialize group with empty root info.";
    }
    return true;
  }
  bool Finalize() override { return true; }
};

///
/// \brief DummyCollectiveCommunicationLib to maintain collective communication relationship without real device
/// communication.
///
class DummyCollectiveCommunicationLib : public CollectiveCommunicationLib {
 public:
  DummyCollectiveCommunicationLib();

  ~DummyCollectiveCommunicationLib() override = default;

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool CreateDeviceCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size) override;

  uint32_t GetRankId(const std::string &group_name) override;

  uint32_t GetGroupSize(const std::string &group_name) override;

  uint32_t GetLocalRankId(const std::string &group_name) override;

  uint32_t GetLocalGroupSize(const std::string &group_name) override;

  uint32_t GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) override;

  uint32_t GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) override;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_DUMMY_COLLECTIVE_COMMUNICATION_LIB_H_
