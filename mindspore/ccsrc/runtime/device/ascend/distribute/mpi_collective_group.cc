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

#include "hccl/hccl.h"
#include "runtime/rt.h"
#include "runtime/device/ascend/distribute/mpi_collective_group.h"
namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
MPICollective::MPICollective() : mpi_inited_(false), rank_id_(0), local_rank_id_(0), rank_size_(0) {}
void MPICollective::FinalizeMPI() {
  group_info_.clear();
  group_comm_.clear();
  int finalized;
  (void)MPI_Finalized(&finalized);
  if (finalized == 0) {
    (void)MPI_Finalize();
  }
}
void MPICollective::DestroyHcclComm() {
  for (auto &it : group_comm_) {
    CHECK_RET(HcclCommDestroy(it.second), ::HcclResult::HCCL_SUCCESS, "HcclCommDestroy failed");
  }
}
MPICollective &MPICollective::instance() {
  static MPICollective instance = {};
  return instance;
}
int MPICollective::GetRankIdByGroup(const std::string &name) {
  CHECK_RET(group_info_.count(name), 1, "Failed to get MPI group rank by group name " + name);
  return group_info_[name].first;
}
int MPICollective::GetGroupSize(const std::string &name) {
  CHECK_RET(group_info_.count(name), 1, "Failed to get MPI group size by group name " + name);
  return group_info_[name].second;
}
HcclComm MPICollective::GetGroupComm(const std::string &name) {
  CHECK_RET(group_comm_.count(name), 1, "Failed to get MPI group comm by group name " + name);
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
  group_info_["hccl_world_group"] = {rank_id_, rank_size_};
  mpi_inited_ = true;
  return true;
}

bool MPICollective::CreateCommGroup(const std::string &name, const std::vector<unsigned int> &ranks) {
  CHECK_RET(mpi_inited_, true, "HcclCollectiveGroup has not been inited.");
  CHECK_RET(ranks.empty(), false, "Ranks is empty.");
  std::vector<int> group_ranks(ranks.begin(), ranks.end());
  CHECK_RET(group_comm_.count(name), 0, "Group comm has already been created.");
  CHECK_RET(rtSetDevice(local_rank_id_), RT_ERROR_NONE, "Call rtSetDevice error.");
  HcclRootInfo rootInfo;
  if (static_cast<unsigned int>(rank_id_) == ranks[0]) {
    CHECK_RET(HcclGetRootInfo(&rootInfo), ::HcclResult::HCCL_SUCCESS, "HcclGetRootInfo failed.");
  }
  MPI_Group mpi_group = MPI_GROUP_NULL;
  CHECK_RET(MPI_Group_incl(comm_group_world_, group_ranks.size(), group_ranks.data(), &mpi_group), MPI_SUCCESS,
            "Create mpi group failed!");
  MPI_Comm mpi_group_comm;

  CHECK_RET(MPI_Comm_create(MPI_COMM_WORLD, mpi_group, &mpi_group_comm), MPI_SUCCESS, "Create mpi comm fail!");

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

  CHECK_RET(HcclCommInitRootInfo(static_cast<uint32_t>(ranks.size()), &rootInfo, static_cast<uint32_t>(group_rank[0]),
                                 &group_hcomm),
            ::HcclResult::HCCL_SUCCESS, "HcclCommInitRootInfo failed.");
  group_comm_[name] = group_hcomm;
  group_info_[name] = {group_rank[0], static_cast<int>(ranks.size())};
  return true;
}
void MPICollective::AssignLocalRankID() {
  char host_name[MAX_HOSTNAME_LEN] = {0};
  CHECK_RET(gethostname(host_name, MAX_HOSTNAME_LEN), MPI_SUCCESS, "Getting host name failed!");
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
