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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_NVIDIA_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_NVIDIA_COMMUNICATION_GROUP_H_

#include <nccl.h>
#include <string>
#include <vector>
#include "runtime/hardware/communication_group.h"

namespace mindspore {
namespace device {
namespace gpu {
class NvidiaCommunicationGroup : public CommunicationGroup {
 public:
  explicit NvidiaCommunicationGroup(const std::string name, const std::vector<uint32_t> &group_ranks)
      : CommunicationGroup(name, group_ranks) {}

  ~NvidiaCommunicationGroup() override = default;

  void Initialize() override;
  void Finalize() override;

 private:
  // The NCCL unique id for this group. Used to initialize this group's communicator.
  ncclUniqueId unique_id_;

  // NCCL communicator of this group.
  ncclComm_t comm_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_NVIDIA_COMMUNICATION_GROUP_H_
