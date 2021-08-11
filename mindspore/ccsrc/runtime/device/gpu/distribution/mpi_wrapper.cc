/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/gpu/distribution/mpi_wrapper.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include "runtime/device/gpu/distribution/nccl_wrapper.h"

namespace mindspore {
namespace device {
namespace gpu {
MPIWrapper::MPIWrapper() : rank_id_(0), rank_size_(0), local_rank_id_(0) { Init(); }

MPIWrapper::~MPIWrapper() {
  int finalized;
  MPI_Finalized(&finalized);
  if (finalized == 0) {
    MPI_Finalize();
  }
}

MPIWrapper &MPIWrapper::instance() {
  static MPIWrapper instance;
  return instance;
}

int MPIWrapper::local_rank_id() const { return local_rank_id_; }

bool MPIWrapper::CreateCommGroup(const std::string &group_name, const std::vector<unsigned int> &group_ranks) {
  std::vector<int> ranks(group_ranks.begin(), group_ranks.end());
  MPI_Group mpi_group;
  CHECK_RET(MPI_Group_incl(world_group_, ranks.size(), ranks.data(), &mpi_group), MPI_SUCCESS,
            "Failed to produce a new group from MPI_COMM_WORLD group for " + group_name);
  SetGroupNameToMPIGroup(group_name, mpi_group);

  MPI_Comm mpi_group_comm;
  CHECK_RET(MPI_Comm_create(MPI_COMM_WORLD, mpi_group, &mpi_group_comm), MPI_SUCCESS,
            "Failed to create MPI communicator.");
  if (mpi_group_comm == MPI_COMM_NULL) {
    return false;
  }

  ncclUniqueId group_unique_id;
  if (ranks.size() == 0) {
    return false;
  }
  if (rank_id_ == ranks[0]) {
    group_unique_id = NCCLWrapper::instance().nccl_unique_id();
  }
  MPI_Bcast(&group_unique_id, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_group_comm);

  int group_rank[1];
  int global_rank[1] = {rank_id_};
  CHECK_RET(MPI_Group_translate_ranks(world_group_, 1, global_rank, mpi_group, group_rank), MPI_SUCCESS,
            "Failed to translate global rank to group rank.");
  if (group_rank[0] == MPI_UNDEFINED) {
    return false;
  }

  NcclGroupInfo nccl_group = {static_cast<int>(ranks.size()), group_rank[0], group_unique_id, nullptr, ranks};
  NCCLWrapper::instance().AddGroupInfo(group_name, &nccl_group);
  return true;
}

int MPIWrapper::GetRankIDByGroup(const std::string &group_name) {
  CHECK_RET(group_name_to_mpi_group_map_.count(group_name), 1, "Failed to get MPI group by group name " + group_name);
  MPI_Group mpi_group = group_name_to_mpi_group_map_[group_name];
  int rank;
  CHECK_RET(MPI_Group_rank(mpi_group, &rank), MPI_SUCCESS, "Failed to get rank id by group name." + group_name);
  return rank;
}

int MPIWrapper::GetGroupSize(const std::string &group_name) {
  CHECK_RET(group_name_to_mpi_group_map_.count(group_name), 1, "Failed to get MPI group by group name" + group_name);
  MPI_Group mpi_group = group_name_to_mpi_group_map_[group_name];
  int size;
  CHECK_RET(MPI_Group_size(mpi_group, &size), MPI_SUCCESS, "Failed to get group size by group name." + group_name);
  return size;
}

bool MPIWrapper::DestroyGroup(const std::string &group_name) {
  auto group_iter = group_name_to_mpi_group_map_.find(group_name);
  if (group_iter == group_name_to_mpi_group_map_.end()) {
    return false;
  }
  group_name_to_mpi_group_map_.erase(group_name);
  MPI_Group mpi_group = group_iter->second;
  CHECK_RET(MPI_Group_free(&mpi_group), MPI_SUCCESS, "Failed to free MPI group for " + group_name);
  NCCLWrapper::instance().DestroyGroup(group_name);
  return true;
}

void MPIWrapper::Init() {
  int initialized;
  CHECK_RET(MPI_Initialized(&initialized), MPI_SUCCESS, "Failed to check mpi initialization status.");
  if (initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  CHECK_RET(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id_), MPI_SUCCESS, "Failed to init mpi rank id.");
  CHECK_RET(MPI_Comm_size(MPI_COMM_WORLD, &rank_size_), MPI_SUCCESS, "Failed to init mpi rank size.");
  AssignLocalRankID();

  CHECK_RET(MPI_Comm_group(MPI_COMM_WORLD, &world_group_), MPI_SUCCESS, "Failed to get group of MPI_COMM_WORLD");
  SetGroupNameToMPIGroup(NCCL_WORLD_GROUP, world_group_);

  ncclUniqueId unique_id;
  if (rank_id_ == 0) {
    unique_id = NCCLWrapper::instance().nccl_unique_id();
  }
  CHECK_RET(MPI_Bcast(reinterpret_cast<void *>(&unique_id), sizeof(unique_id), MPI_BYTE, 0, MPI_COMM_WORLD),
            MPI_SUCCESS, "Failed to broadcast nccl unique id.");

  std::vector<int> world_group_ranks = {};
  for (int global_rank = 0; global_rank < rank_size_; global_rank++) {
    world_group_ranks.push_back(global_rank);
  }
  NcclGroupInfo world_group = {rank_size_, rank_id_, unique_id, nullptr, world_group_ranks};
  NCCLWrapper::instance().AddGroupInfo(NCCL_WORLD_GROUP, &world_group);
  return;
}

void MPIWrapper::AssignLocalRankID() {
  char host_name[MAX_HOSTNAME_LEN] = {0};
  CHECK_RET(gethostname(host_name, MAX_HOSTNAME_LEN), 0, "Getting host name failed.");
  size_t host_hash = std::hash<std::string>()(host_name);

  const int kRankSize = rank_size_;
  size_t all_host_hashs[kRankSize];
  CHECK_RET((rank_id_ < kRankSize), true, "The rank id is not less than rank size.");
  all_host_hashs[rank_id_] = host_hash;
  CHECK_RET(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_host_hashs, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD),
            MPI_SUCCESS, "MPI_Allgather host hashes failed.");
  for (int global_rank = 0; global_rank < kRankSize; global_rank++) {
    if (global_rank == rank_id_) {
      break;
    }
    if (all_host_hashs[global_rank] == all_host_hashs[rank_id_]) {
      local_rank_id_++;
    }
  }
  return;
}

void MPIWrapper::SetGroupNameToMPIGroup(const std::string &group_name, const MPI_Group mpi_group) {
  group_name_to_mpi_group_map_[group_name] = mpi_group;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
