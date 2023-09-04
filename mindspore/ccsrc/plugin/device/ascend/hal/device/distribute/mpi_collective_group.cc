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

#include <algorithm>
#include "hccl/hccl.h"
#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#include "plugin/device/ascend/hal/device/distribute/mpi_collective_group.h"
namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
MPICollective::MPICollective()
    : mpi_inited_(false), rank_id_(0), local_rank_id_(0), rank_size_(0), comm_group_world_(MPI_GROUP_NULL) {}
void MPICollective::FinalizeMPI() {
  group_info_.clear();
  group_comm_.clear();
  int finalized;
  (void)MPI_Finalized(&finalized);
  if (finalized == 0) {
    (void)MPI_Finalize();
  }
}

MPICollective::~MPICollective() {
  int finalized;
  (void)MPI_Finalized(&finalized);
  if (finalized == 0) {
    (void)MPI_Finalize();
  }
}

void MPICollective::DestroyHcclComm() {
  for (auto iter = group_comm_.cbegin(); iter != group_comm_.cend(); ++iter) {
    CHECK_RET(static_cast<int32_t>(HcclCommDestroy(iter->second)), static_cast<int32_t>(::HcclResult::HCCL_SUCCESS),
              "HcclCommDestroy failed");
  }
  group_comm_.clear();
}

MPICollective &MPICollective::instance() {
  static MPICollective instance = {};
  return instance;
}

int MPICollective::GetRankIdByGroup(const std::string &name) {
  CHECK_RET(group_info_.count(name), 1, ("Failed to get MPI group rank by group name " + name));
  return std::get<0>(group_info_[name]);
}

int MPICollective::GetGroupSize(const std::string &name) {
  CHECK_RET(group_info_.count(name), 1, ("Failed to get MPI group size by group name " + name));
  return std::get<1>(group_info_[name]);
}

int MPICollective::GetGroupLocalRankSize(const std::string &name) {
  CHECK_RET(group_info_.count(name), 1, ("Failed to get MPI group local size by group name " + name));
  return std::get<local_rank_size_index>(group_info_[name]);
}

int MPICollective::GetWorldRankIdFromGroup(const std::string &name, const int rank_id) {
  CHECK_RET(world_map_.count(name), 1, ("Failed to get MPI world rank from group by group name " + name));
  CHECK_RET(static_cast<int>(world_map_[name].size()) > rank_id && rank_id >= 0, 1,
            ("The rank_id " + std::to_string(rank_id) + "is not in the range of group " + name));
  CHECK_RET(rank_id >= 0, true, "The rank_id[" + std::to_string(rank_id) + "] must be greater equal than zero.");
  return world_map_[name][static_cast<uint32_t>(rank_id)];
}

int MPICollective::GetGroupRankIdFromWorld(const std::string &name, const int rank_id) {
  CHECK_RET(world_map_.count(name), 1, ("Failed to get MPI group rank from world by group name " + name));
  CHECK_RET(std::min(rank_size_ - 1, rank_id), rank_id,
            ("The rank_id " + std::to_string(rank_id) + "is great than world rank size"));
  CHECK_RET(std::count(world_map_[name].begin(), world_map_[name].end(), rank_id), 1,
            ("The rank_id " + std::to_string(rank_id) + " is not in group " + name));
  return std::find(world_map_[name].begin(), world_map_[name].end(), rank_id) - world_map_[name].begin();
}

HcclComm MPICollective::GetGroupComm(const std::string &name) {
  CHECK_RET(group_comm_.count(name), 1, ("Failed to get MPI group comm by group name " + name));
  return group_comm_[name];
}

int MPICollective::GetDeviceId() const { return local_rank_id_; }

bool MPICollective::Init() {
  int init_flag = 0;
  CHECK_RET(MPI_Initialized(&init_flag), MPI_SUCCESS, "Check mpi initialized fail!");
  if (init_flag == 0) {
    CHECK_RET(MPI_Init(nullptr, nullptr), MPI_SUCCESS, "Failed to init mpi!");
  }

  CHECK_RET(MPI_Comm_group(MPI_COMM_WORLD, &comm_group_world_), MPI_SUCCESS, "comm_group_world_ init fail!");

  CHECK_RET(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id_), MPI_SUCCESS, "Failed to init mpi rank id!");

  CHECK_RET(MPI_Comm_size(MPI_COMM_WORLD, &rank_size_), MPI_SUCCESS, "Failed to init mpi rank size!");
  AssignLocalRankID();
  group_info_["hccl_world_group"] = std::make_tuple(rank_id_, rank_size_, 0);
  mpi_inited_ = true;
  return true;
}

bool MPICollective::CreateCommGroup(const std::string &name, const std::vector<unsigned int> &ranks) {
  CHECK_RET(mpi_inited_, true, "HcclCollectiveGroup has not been inited.");
  CHECK_RET(ranks.empty(), false, "Ranks is empty.");
  std::vector<int> group_ranks(ranks.begin(), ranks.end());
  if (group_comm_.count(name) != 0) {
    return true;
  }
  CHECK_RET(aclrtSetDevice(local_rank_id_), ACL_ERROR_NONE, "Call aclrtSetDevice error.");
  HcclRootInfo rootInfo;
  if (static_cast<unsigned int>(rank_id_) == ranks[0]) {
    CHECK_RET(static_cast<int32_t>(HcclGetRootInfo(&rootInfo)), static_cast<int32_t>(::HcclResult::HCCL_SUCCESS),
              "HcclGetRootInfo failed.");
  }
  MPI_Group mpi_group = MPI_GROUP_NULL;
  CHECK_RET(MPI_Group_incl(comm_group_world_, group_ranks.size(), group_ranks.data(), &mpi_group), MPI_SUCCESS,
            "Create mpi group failed!");
  MPI_Comm mpi_group_comm;

  CHECK_RET(MPI_Comm_create_group(MPI_COMM_WORLD, mpi_group, 0, &mpi_group_comm), MPI_SUCCESS, "Create mpi comm fail!");

  CHECK_RET(MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, mpi_group_comm), MPI_SUCCESS,
            "Mpi reduce_scatter failed!");

  HcclComm group_hcomm = nullptr;
  int group_rank[1];
  int global_rank[1] = {rank_id_};
  CHECK_RET(MPI_Group_translate_ranks(comm_group_world_, 1, global_rank, mpi_group, group_rank), MPI_SUCCESS,
            "Failed to translate global rank to group rank.");
  if (group_rank[0] == MPI_UNDEFINED) {
    return false;
  }

  CHECK_RET(static_cast<int32_t>(HcclCommInitRootInfo(static_cast<uint32_t>(ranks.size()), &rootInfo,
                                                      static_cast<uint32_t>(group_rank[0]), &group_hcomm)),
            static_cast<int32_t>(::HcclResult::HCCL_SUCCESS), "HcclCommInitRootInfo failed.");
  group_comm_[name] = group_hcomm;
  group_info_[name] = std::make_tuple(group_rank[0], static_cast<int>(ranks.size()), 0);
  AssignLocalRankSize(name, group_ranks, mpi_group_comm);
  return true;
}

void MPICollective::AssignLocalRankSize(const std::string &name, const std::vector<int> &group_ranks,
                                        MPI_Comm mpi_group_comm) {
  char host_name[max_hostname_len] = {0};
  CHECK_RET(gethostname(host_name, max_hostname_len), MPI_SUCCESS, "Getting host name failed!");
  size_t host_hash = std::hash<std::string>()(host_name);

  auto rank_size = group_ranks.size();
  std::vector<size_t> all_host_hashs(rank_size);
  for (size_t i = 0; i < rank_size; ++i) {
    if (group_ranks[i] == rank_id_) {
      all_host_hashs[i] = host_hash;
    }
  }
  CHECK_RET(
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_host_hashs.data(), sizeof(size_t), MPI_BYTE, mpi_group_comm),
    MPI_SUCCESS, "MPI_Allgather host hash failed.");
  int local_rank_size = static_cast<int>(std::count(all_host_hashs.begin(), all_host_hashs.end(), host_hash));
  std::get<local_rank_size_index>(group_info_[name]) = local_rank_size;
  std::vector<int> group_world_ranks(group_ranks.begin(), group_ranks.end());
  world_map_[name] = group_world_ranks;
}

void MPICollective::AssignLocalRankID() {
  char host_name[max_hostname_len] = {0};
  CHECK_RET(gethostname(host_name, max_hostname_len), MPI_SUCCESS, "Getting host name failed!");
  size_t host_hash = std::hash<std::string>()(host_name);

  const int kRankSize = rank_size_;
  size_t all_host_hashs[kRankSize];
  all_host_hashs[rank_id_] = host_hash;
  CHECK_RET(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_host_hashs, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD),
            MPI_SUCCESS, "MPI_Allgather host hash failed.");
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
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
