/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COMMUNICATION_GROUP_H_

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <unordered_map>
#include "hccl/hccl.h"
#include "include/runtime/hardware_abstract/collective/communication_group.h"

namespace mindspore {
namespace device {
namespace ascend {
// This the config passed to 'CreateCommunicationGroup' method. It controls initialization mode for communication group.
constexpr char kHCCLGlobalGroupName[] = "hccl_world_group";
// Confirmed by HCCL max length of hccl comm name is 128.
constexpr int INNER_COMM_NAME_MAX_LENGTH = 128;

class AscendCommunicationGroup : public CommunicationGroup {
 public:
  explicit AscendCommunicationGroup(
    const std::string &name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank, uint32_t local_group_rank,
    uint32_t local_group_size,
    const std::unordered_map<std::string, std::variant<int64_t, uint32_t, std::string>> &hccl_config = {});

  ~AscendCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  // Check if CANN so supports the target HCCL comm config parameters.
  bool IsSupportConfigParameter(uint32_t config_parameter);

  // Return HcclCommConfig based on users' options configuration.
  HcclCommConfig CreateHcclCommConfig();

  // Return HCCL communicator because collective operations need it as a input.
  const HcclComm &hccl_communicator() const;

  // Return communicator name maintained by HCCL. This is different from the group set by user.
  std::string inner_comm_name() const;

 private:
  // Initialize HCCL communicator by root info, using API HcclCommInitRootInfoConfig.
  bool InitByRootInfoConfig(void *root_info, uint32_t group_size, uint32_t group_rank, const HcclCommConfig &config);

  // Initialize HCCL communicator by rank table if the rank table is configured. Note that HCCL initialization APIs
  // for global_comm (HcclCommInitClusterInfoConfig) and sub_comm (HcclCreateSubCommConfig) are different when using
  // rank table.
  bool InitByRankTable(std::string rank_table, uint32_t group_size, uint32_t group_rank, HcclCommConfig *config);

  bool InitByHcclComm();

  // Initialpize HCCL config parameters, such as hcclBufferSize and hcclDeterministic.
  void InitHcclCommConfig(HcclCommConfig *config);

  // Set JobId and NodeId to HCCL comm config for nslb.
  void SetNslbCommConfig(HcclCommConfig *config);

  // The HCCL unique id for this group. Used to initialize this group's communicator.
  HcclRootInfo unique_id_;

  // HCCL communicator of this group.
  HcclComm comm_;

  // hccl_config pass from previous GroupOptions.
  std::unordered_map<std::string, std::variant<int64_t, uint32_t, std::string>> hccl_config_;

  char inner_comm_name_[INNER_COMM_NAME_MAX_LENGTH];
};
using AscendCommunicationGroupPtr = std::shared_ptr<AscendCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COMMUNICATION_GROUP_H_
