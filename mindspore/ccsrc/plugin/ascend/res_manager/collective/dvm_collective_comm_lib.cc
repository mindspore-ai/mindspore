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

#include "plugin/ascend/res_manager/collective/dvm_collective_comm_lib.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;

DvmCollectiveCommLib &DvmCollectiveCommLib::GetInstance() {
  static DvmCollectiveCommLib instance;
  return instance;
}

DvmCollectiveCommLib::DvmCollectiveCommLib() { global_group_name_ = kHCCLWorldGroup; }

bool DvmCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool DvmCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                    const std::vector<uint32_t> &group_ranks, uint32_t local_group_rank,
                                                    uint32_t local_group_size, const GroupOptions &config) {
  CHECK_RET((groups_.count(group_name) == 0), true,
            "The DVM communication group " + group_name + " has already existed.");

  DvmCommunicationGroupPtr group = std::make_shared<DvmCommunicationGroup>(group_name, group_ranks, global_rank_id_,
                                                                           local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

CommPtr DvmCollectiveCommLib::GetCommunicator(const std::string &group_name) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The DVM communication group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<DvmCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->dvm_communicator();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
