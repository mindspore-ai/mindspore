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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_COLLECTIVE_DVM_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_COLLECTIVE_DVM_COLLECTIVE_COMM_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "plugin/ascend/res_manager/collective/dvm_communication_group.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;

class EXPORT_WRAPPER DvmCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static DvmCollectiveCommLib &GetInstance();

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size,
                                const GroupOptions &config = {}) override;

  CommPtr GetCommunicator(const std::string &group_name);

 private:
  DvmCollectiveCommLib();
  ~DvmCollectiveCommLib() override = default;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_COLLECTIVE_DVM_COLLECTIVE_COMM_LIB_H_
