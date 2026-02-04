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
#include "include/runtime/hardware_abstract/collective/dummy_collective_communication_lib.h"
#include <algorithm>
#include <numeric>
#include <memory>
#include "include/utils/utils.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
DummyCollectiveCommunicationLib::DummyCollectiveCommunicationLib() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    global_group_name_ = kHCCLWorldGroup;
  } else {
    global_group_name_ = kNCCLWorldGroup;
  }
}

bool DummyCollectiveCommunicationLib::Initialize(uint32_t global_rank, uint32_t global_rank_size,
                                                 uint32_t local_rank_id) {
  if (initialized_) {
    return true;
  }

  global_rank_size_ = global_rank_size;
  global_rank_id_ = global_rank;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool DummyCollectiveCommunicationLib::CreateDeviceCommunicationGroup(const std::string &group_name,
                                                                     const std::vector<uint32_t> &group_ranks) {
  auto default_group_size = GetLocalGroupSize(group_name);
  if (default_group_size == 0) {
    default_group_size = kDefaultLocalRankSize;
  }
  auto rank_id = GetRankId(group_name);
  uint32_t node_num = rank_id / default_group_size;
  uint32_t node_rank_low = node_num * default_group_size;
  uint32_t node_rank_high = (node_num + 1) * default_group_size;
  auto local_group_size =
    static_cast<uint32_t>(std::count_if(group_ranks.begin(), group_ranks.end(),
                                        [&](uint32_t rank) { return rank >= node_rank_low && rank < node_rank_high; }));

  auto pos = find(group_ranks.begin(), group_ranks.end(), rank_id);
  uint32_t local_group_rank = 0;
  if (pos == group_ranks.end()) {
    local_group_rank = UINT32_MAX;
  } else {
    local_group_rank = static_cast<uint32_t>(std::count_if(
      group_ranks.begin(), pos, [&](uint32_t rank) { return rank >= node_rank_low && rank < node_rank_high; }));
  }
  return CreateCommunicationGroup(group_name, group_ranks, local_group_rank, local_group_size);
}

bool DummyCollectiveCommunicationLib::CreateCommunicationGroup(const std::string &group_name,
                                                               const std::vector<uint32_t> &group_ranks,
                                                               uint32_t local_group_rank, uint32_t local_group_size,
                                                               const GroupOptions &config) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(WARNING) << "The group " << group_name << " has already existed.";
    return true;
  }
  auto group = std::make_shared<DummyCommunicationGroup>(group_name, group_ranks, GetRankId(group_name),
                                                         local_group_rank, local_group_size);
  groups_[group_name] = group;
  return true;
}

uint32_t DummyCollectiveCommunicationLib::GetRankId(const std::string &group_name) {
  if (!common::GetEnv(kEnvRankId).empty()) {
    global_rank_id_ = LongToUint(std::strtol(common::GetEnv(kEnvRankId).c_str(), nullptr, kDecimalBase));
  }

  if (groups_.count(group_name) != 0) {
    return CollectiveCommunicationLib::GetRankId(group_name);
  }
  return global_rank_id_;
}

uint32_t DummyCollectiveCommunicationLib::GetGroupSize(const std::string &group_name) {
  if (!common::GetEnv(kEnvRankSize).empty()) {
    global_rank_size_ = LongToUint(std::strtol(common::GetEnv(kEnvRankSize).c_str(), nullptr, kDecimalBase));
  }

  if (groups_.count(group_name) != 0) {
    return CollectiveCommunicationLib::GetGroupSize(group_name);
  }
  return global_rank_size_;
}

uint32_t DummyCollectiveCommunicationLib::GetLocalRankId(const std::string &group_name) {
  if (groups_.count(group_name) != 0) {
    return CollectiveCommunicationLib::GetLocalRankId(group_name);
  }
  if (!common::GetEnv(kEnvLocalRankId).empty()) {
    return LongToUint(std::strtol(common::GetEnv(kEnvLocalRankId).c_str(), nullptr, kDecimalBase));
  }
  return GetRankId(group_name) % kDefaultLocalRankSize;
}

uint32_t DummyCollectiveCommunicationLib::GetLocalGroupSize(const std::string &group_name) {
  if (groups_.count(group_name) != 0) {
    return CollectiveCommunicationLib::GetLocalGroupSize(group_name);
  }
  if (!common::GetEnv(kEnvLocalRankSize).empty()) {
    return LongToUint(std::strtol(common::GetEnv(kEnvLocalRankSize).c_str(), nullptr, kDecimalBase));
  }
  auto rank_size = GetGroupSize(group_name);
  if (rank_size < kDefaultLocalRankSize) {
    return rank_size;
  }
  return kDefaultLocalRankSize;
}

uint32_t DummyCollectiveCommunicationLib::GetWorldRankFromGroupRank(const std::string &group_name,
                                                                    uint32_t local_rank) {
  if (groups_.count(group_name) != 0) {
    return CollectiveCommunicationLib::GetWorldRankFromGroupRank(group_name, local_rank);
  }
  return local_rank;
}

uint32_t DummyCollectiveCommunicationLib::GetGroupRankFromWorldRank(uint32_t world_rank,
                                                                    const std::string &group_name) {
  if (groups_.count(group_name) != 0) {
    return CollectiveCommunicationLib::GetGroupRankFromWorldRank(world_rank, group_name);
  }
  return world_rank;
}
}  // namespace device
}  // namespace mindspore
