/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <mpi.h>
#include <string>
#include <vector>
#include <memory>
#include "hccl/hccl.h"
#include "runtime/collective/communication_group.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendCommunicationGroup : public CommunicationGroup {
 public:
  explicit AscendCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                    uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~AscendCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  // Return HCCL communicator because collective operations need it as a input.
  const HcclComm &hccl_communicator() const;

 private:
  // The HCCL unique id for this group. Used to initialize this group's communicator.
  HcclRootInfo unique_id_;

  // HCCL communicator of this group.
  HcclComm comm_;
};
using AscendCommunicationGroupPtr = std::shared_ptr<AscendCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COMMUNICATION_GROUP_H_
