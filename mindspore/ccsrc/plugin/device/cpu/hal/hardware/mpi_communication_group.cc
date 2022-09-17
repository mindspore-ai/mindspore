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

#include "plugin/device/cpu/hal/hardware/mpi_communication_group.h"

namespace mindspore {
namespace device {
namespace cpu {
MPICommunicationGroup::MPICommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                             uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      group_(MPI_GROUP_NULL),
      group_communicator_(MPI_COMM_NULL) {}

bool MPICommunicationGroup::Finalize() {
  if (!initialized_) {
    return false;
  }

  CHECK_RET(MPI_Comm_free(&group_communicator_), MPI_SUCCESS,
            "Freeing MPI group communicator for " + name_ + " failed.");
  CHECK_RET(MPI_Group_free(&group_), MPI_SUCCESS, "Freeing MPI group for " + name_ + " failed.");
  initialized_ = false;
  return true;
}

bool MPICommunicationGroup::Initialize(const MPI_Group &world_group) {
  if (initialized_) {
    return false;
  }
  std::vector<int> ranks(group_ranks_.begin(), group_ranks_.end());
  CHECK_RET(MPI_Group_incl(world_group, static_cast<int>(ranks.size()), ranks.data(), &group_), MPI_SUCCESS,
            "Creating MPI group for " + name_ + " failed.");
  CHECK_RET(MPI_Comm_create_group(MPI_COMM_WORLD, group_, 0, &group_communicator_), MPI_SUCCESS,
            "Creating MPI group communicator for " + name_ + " failed.");

  CHECK_RET((group_communicator_ != MPI_COMM_NULL), true, "Failed to create MPI communicator for group " + name_);
  initialized_ = true;
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
