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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_MPI_COLLECTIVE_INIT_H
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_MPI_COLLECTIVE_INIT_H

#include <mpi.h>
#include <map>
#include <tuple>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include "hccl/hccl_types.h"
#include "pybind11/pybind11.h"
namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
constexpr int max_hostname_len = 1024;
constexpr int local_rank_size_index = 2;
class MPICollective {
 public:
  MPICollective(MPICollective const &) = delete;
  MPICollective &operator=(const MPICollective &) = delete;
  static MPICollective &instance();
  void AssignLocalRankID();
  void AssignLocalRankSize();
  bool Init();
  void FinalizeMPI();
  int GetRankIdByGroup(const std::string &name);
  int GetGroupSize(const std::string &name);
  int GetGroupLocalRankSize(const std::string &name);
  int GetWorldRankIdFromGroup(const std::string &name, const int rank_id);
  int GetGroupRankIdFromWorld(const std::string &name, const int rank_id);
  void AssignLocalRankSize(const std::string &name, const std::vector<int> &group_ranks, MPI_Comm mpi_group_comm);
  HcclComm GetGroupComm(const std::string &name);
  int GetDeviceId() const;
  bool CreateCommGroup(const std::string &name, const std::vector<unsigned int> &ranks);
  void DestroyHcclComm();
  std::map<std::string, HcclComm> group_comm_;

 private:
  MPICollective();
  ~MPICollective();
  bool mpi_inited_;
  int rank_id_;
  int local_rank_id_;
  int rank_size_;
  MPI_Group comm_group_world_;
  std::map<std::string, std::tuple<int, int, int>> group_info_;
  std::map<std::string, std::vector<int>> world_map_;
};
#define CHECK_RET(expression, result, message)                                         \
  {                                                                                    \
    auto ret = (expression);                                                           \
    if (ret != (result)) {                                                             \
      std::ostringstream oss;                                                          \
      oss << "Error in file " << __FILE__ << " | Error on line " << __LINE__           \
          << " | Ascend collective Error: " << (message) << " | Error Number " << ret; \
      pybind11::pybind11_fail(oss.str());                                              \
    }                                                                                  \
  }
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_COLLECTIVE_INIT_H
