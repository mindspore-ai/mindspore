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

#include "runtime/hardware/gpu/nvidia_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace gpu {
NvidiaCollectiveCommLib::NvidiaCollectiveCommLib() { global_group_name_ = kNCCLGlobalGroupName; }

bool NvidiaCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  initialized_ = true;
  return true;
}

bool NvidiaCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                       const std::vector<uint32_t> &group_ranks) {
  CHECK_RET((groups_.count(group_name) == 0), true, "The NCCL group " + group_name + " has already existed.");

  NvidiaCommunicationGroupPtr group =
    std::make_shared<NvidiaCommunicationGroup>(group_name, group_ranks, global_rank_id_);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}
}  // namespace gpu

using NvidiaCollectiveCommLib = mindspore::device::gpu::NvidiaCollectiveCommLib;
CollectiveCommunicationLib *communication_lib_instance() { return &NvidiaCollectiveCommLib::GetInstance(); }
}  // namespace device
}  // namespace mindspore
