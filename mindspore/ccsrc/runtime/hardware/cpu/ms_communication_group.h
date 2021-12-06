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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MS_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MS_COMMUNICATION_GROUP_H_

#include <string>
#include <vector>
#include <memory>
#include "runtime/hardware/collective/communication_group.h"

namespace mindspore {
namespace device {
namespace cpu {
class MsCommunicationGroup : public CommunicationGroup {
 public:
  explicit MsCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank)
      : CommunicationGroup(name, group_ranks, global_rank) {}

  ~MsCommunicationGroup() override = default;

  bool Initialize(void *root_info) override { return true; }
  bool Finalize() override { return true; }
};
using MsCommunicationGroupPtr = std::shared_ptr<MsCommunicationGroup>;
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MS_COMMUNICATION_GROUP_H_
