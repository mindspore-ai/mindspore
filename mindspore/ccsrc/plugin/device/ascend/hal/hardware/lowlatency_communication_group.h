/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LOWLATENCY_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LOWLATENCY_COMMUNICATION_GROUP_H_

#include <string>
#include <vector>
#include <memory>
#include "lccl.h"
#include "lcoc.h"
#include "runtime/collective/communication_group.h"
#include "utils/dlopen_macro.h"

using namespace Lcal;
using LcalCommPtr = std::shared_ptr<LcalComm>;
using LcclPtr = std::shared_ptr<Lccl>;
using LcocPtr = std::shared_ptr<Lcoc>;

namespace mindspore {
namespace device {
namespace ascend {

class LowlatencyCommunicationGroup : public CommunicationGroup {
 public:
  explicit LowlatencyCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                        uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~LowlatencyCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  // Return communicator for collective communication ops.
  const LcclPtr &lccl_communicator() const;
  // Return communicator of lcal.
  const LcalCommPtr &lcal_comm() const;

 private:
  // Lcal communicator of this group, but this should be encapsulated by 'Lccl' class to use communication operations.
  LcalCommPtr lcal_comm_;

  // 'Lccl' object returned to call communication operations.
  LcclPtr lccl_comm_;
};
using LowlatencyCommunicationGroupPtr = std::shared_ptr<LowlatencyCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_LOWLATENCY_COMMUNICATION_GROUP_H_
