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

#include "plugin/device/cpu/hal/hardware/mpi_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace cpu {
MPICollectiveCommLib::MPICollectiveCommLib() : world_group_(MPI_GROUP_NULL) {
  global_group_name_ = kMPIGlobalGroupName;
}

MPICollectiveCommLib::~MPICollectiveCommLib() {
  int finalized;
  (void)MPI_Finalized(&finalized);
  if (finalized == 0) {
    (void)MPI_Finalize();
  }
}

bool MPICollectiveCommLib::Initialize(uint32_t, uint32_t, uint32_t) {
  if (initialized_) {
    return false;
  }

  int initialized = 0;
  CHECK_RET(MPI_Initialized(&initialized), MPI_SUCCESS, "Failed to check MPI initialization status.");
  if (initialized == 0) {
    CHECK_RET(MPI_Init(nullptr, nullptr), MPI_SUCCESS, "Failed to initialize MPI.");
  }

  // Generated MPI global rank id and rank size for the world group MPI_COMM_WORLD.
  int rank_id = 0;
  int rank_size = 0;
  CHECK_RET(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id), MPI_SUCCESS, "Failed to initialize MPI global rank id.");
  CHECK_RET(MPI_Comm_size(MPI_COMM_WORLD, &rank_size), MPI_SUCCESS, "Failed to initialize MPI global rank size.");
  global_rank_id_ = static_cast<uint32_t>(rank_id);
  global_rank_size_ = static_cast<uint32_t>(rank_size);

  // Create the world group of MPI because every other group is generated from MPI world group.
  CHECK_RET(MPI_Comm_group(MPI_COMM_WORLD, &world_group_), MPI_SUCCESS, "Failed to get group of MPI_COMM_WORLD.");
  initialized_ = true;
  return true;
}

bool MPICollectiveCommLib::AllGatherHostHashName(size_t, std::vector<size_t> *host_hash_names) const {
  CHECK_IF_NULL(host_hash_names);
  CHECK_RET(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hash_names->data(), sizeof(size_t), MPI_BYTE,
                          MPI_COMM_WORLD),
            MPI_SUCCESS, "MPI_Allgather host hashes failed.");
  return true;
}

bool MPICollectiveCommLib::BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The MPI group " + group_name + " does not exist.");

  auto mpi_group = std::dynamic_pointer_cast<MPICommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(mpi_group);

  const auto &group_comm = mpi_group->mpi_communicator();
  CHECK_RET(MPI_Bcast(root_info, root_info_size, MPI_BYTE, 0, group_comm), MPI_SUCCESS,
            "Failed to broadcast nccl unique id.");
  return true;
}

bool MPICollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                    const std::vector<uint32_t> &group_ranks, uint32_t local_group_rank,
                                                    uint32_t local_group_size) {
  CHECK_RET((groups_.count(group_name) == 0), true, "The MPI group " + group_name + " has already existed.");

  MPICommunicationGroupPtr group = std::make_shared<MPICommunicationGroup>(group_name, group_ranks, global_rank_id_,
                                                                           local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  CHECK_RET(group->Initialize(world_group_), true, "Failed to initialize group " + group_name);
  groups_[group_name] = group;
  return true;
}
}  // namespace cpu

using MPICollectiveCommLib = mindspore::device::cpu::MPICollectiveCommLib;
CollectiveCommunicationLib *communication_lib_instance() { return &MPICollectiveCommLib::GetInstance(); }
}  // namespace device
}  // namespace mindspore
