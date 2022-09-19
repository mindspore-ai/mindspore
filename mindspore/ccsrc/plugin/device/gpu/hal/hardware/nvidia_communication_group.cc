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

#include "plugin/device/gpu/hal/hardware/nvidia_communication_group.h"

namespace mindspore {
namespace device {
namespace gpu {
NvidiaCommunicationGroup::NvidiaCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                                   uint32_t global_rank, uint32_t local_group_rank,
                                                   uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      unique_id_({}),
      comm_(nullptr) {}

bool NvidiaCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return false;
  }

  // The unique id is broadcasted by the root rank.
  unique_id_ = *(static_cast<ncclUniqueId *>(root_info));
  uint32_t group_rank = GetGroupRank(global_rank_);
  // Initialize the NCCL communicator while the group created. Pay attention that 'ncclCommInitRank' should be called
  // after GPU device id is set.
  CHECK_RET(ncclCommInitRank(&comm_, static_cast<int>(size_), unique_id_, static_cast<int>(group_rank)), ncclSuccess,
            "Initializing NCCL communicator failed.");
  initialized_ = true;
  return true;
}

bool NvidiaCommunicationGroup::Finalize() {
  if (!initialized_) {
    return false;
  }

  // Finalize could be called after any exception is thrown. So we use 'ncclCommAbort' instead of 'ncclCommDestroy'
  // because 'ncclCommAbort' will abort any uncompleted operations before destroying the communicator, e.g.,
  // ncclAllReduce.
  CHECK_RET(ncclCommAbort(comm_), ncclSuccess, "Failed to abort NCCL communicator.");
  initialized_ = false;
  return true;
}

void *NvidiaCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  *root_info_size = sizeof(unique_id_);
  uint32_t group_rank = GetGroupRank(global_rank_);
  if (group_rank == 0) {
    CHECK_RET(ncclGetUniqueId(&unique_id_), ncclSuccess, "Failed to get NCCL unique id.");
  }
  return &unique_id_;
}

const ncclComm_t &NvidiaCommunicationGroup::nccl_communicator() const { return comm_; }
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
