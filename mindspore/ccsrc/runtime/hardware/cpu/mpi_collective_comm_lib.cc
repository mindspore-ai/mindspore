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

#include "runtime/hardware/cpu/mpi_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace cpu {
void MPICollectiveCommLib::Initialize(uint32_t) {
  int initialized = 0;
  CHECK_MPI_RET(MPI_Initialized(&initialized), "Failed to check MPI initialization status.");
  if (initialized == 0) {
    CHECK_MPI_RET(MPI_Init(nullptr, nullptr), "Failed to initialize MPI.");
  }

  // Generated MPI global rank id and rank size for the world group MPI_COMM_WORLD.
  int rank_id = 0;
  int rank_size = 0;
  CHECK_MPI_RET(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id), "Failed to initialize MPI global rank id.");
  CHECK_MPI_RET(MPI_Comm_rank(MPI_COMM_WORLD, &rank_size), "Failed to initialize MPI global rank size.");
  global_rank_id_ = IntToUint(rank_id);
  global_rank_size_ = IntToUint(rank_size);
  MS_LOG(INFO) << "The MPI global rank id of this process is " << global_rank_id_ << ", global rank size is "
               << global_rank_size_;

  // Create the world group of MPI because every other group is generated from MPI world group.
  CHECK_MPI_RET(MPI_Comm_group(MPI_COMM_WORLD, &world_group_), "Failed to get group of MPI_COMM_WORLD.");
}

void MPICollectiveCommLib::Finalize() {
  // The world group is also stored in groups_. So we don't need to finalize world group separately.
  for (const auto &group : groups_) {
    MS_EXCEPTION_IF_NULL(group.second);
    group.second->Finalize();
  }
  groups_.clear();
}

bool MPICollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                    const std::vector<uint32_t> &group_ranks) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(EXCEPTION) << "The MPI group " << group_name << " has already existed.";
    return false;
  }

  MPICommunicationGroupPtr group = std::make_shared<MPICommunicationGroup>(group_name, group_ranks);
  MS_EXCEPTION_IF_NULL(group);
  group->Initialize(world_group_);
  groups_[group_name] = group;
  MS_LOG(INFO) << "MPI group of " << group_name << " is created.";
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
