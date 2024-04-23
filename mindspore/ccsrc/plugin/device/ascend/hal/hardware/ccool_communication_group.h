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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_CCOOL_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_CCOOL_COMMUNICATION_GROUP_H_

#include <memory>
#include <vector>
#include <string>

namespace mindspore {
namespace device {
namespace ascend {
class CcoolCommunicationGroup : public CommunicationGroup {
 public:
  explicit CcoolCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                   uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~CcoolCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;
};
using CcoolCommunicationGroupPtr = std::shared_ptr<CcoolCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_CCOOL_COMMUNICATION_GROUP_H_
