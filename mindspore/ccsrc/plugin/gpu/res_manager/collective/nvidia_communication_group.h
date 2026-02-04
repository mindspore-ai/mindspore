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
#include <memory>
#include "include/runtime/hardware_abstract/collective/communication_group.h"

namespace mindspore {
namespace device {
namespace gpu {
class NvidiaCommunicationGroup : public CommunicationGroup {
 public:
  explicit NvidiaCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                    uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~NvidiaCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  // Return NCCL communicator because collective operations need it as a input.
  const ncclComm_t &nccl_communicator() const;

 private:
  // The NCCL unique id for this group. Used to initialize this group's communicator.
  ncclUniqueId unique_id_;

  // NCCL communicator of this group.
  ncclComm_t comm_;
};
using NvidiaCommunicationGroupPtr = std::shared_ptr<NvidiaCommunicationGroup>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_NVIDIA_COMMUNICATION_GROUP_H_
