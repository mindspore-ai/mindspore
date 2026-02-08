/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_COLLECTIVE_DVM_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_COLLECTIVE_DVM_COMMUNICATION_GROUP_H_

#include <string>
#include <vector>
#include <memory>
#include "kernel/ascend/dvm/dvm.h"
#include "include/runtime/hardware_abstract/collective/communication_group.h"

using CommPtr = std::shared_ptr<dvm::Comm>;

namespace mindspore {
namespace device {
namespace ascend {

class DvmCommunicationGroup : public CommunicationGroup {
 public:
  explicit DvmCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                 uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~DvmCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  // Return communicator for collective communication ops.
  const CommPtr &dvm_communicator() const;

 private:
  CommPtr dvm_comm_;
};
using DvmCommunicationGroupPtr = std::shared_ptr<DvmCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_COLLECTIVE_DVM_COMMUNICATION_GROUP_H_
