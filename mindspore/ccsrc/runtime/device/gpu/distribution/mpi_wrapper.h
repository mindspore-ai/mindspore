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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_MPI_WRAPPER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_MPI_WRAPPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "runtime/device/gpu/distribution/collective_common.h"

namespace mindspore {
namespace device {
namespace gpu {
class MPIWrapper {
 public:
  MPIWrapper(MPIWrapper const &) = delete;
  MPIWrapper &operator=(const MPIWrapper &) = delete;
  static MPIWrapper &instance();
  int local_rank_id() const;
  bool CreateCommGroup(const std::string &group_name, const std::vector<unsigned int> &ranks);
  int GetRankIDByGroup(const std::string &group_name);
  int GetGroupSize(const std::string &group_name);
  bool DestroyGroup(const std::string &group_name);

 private:
  MPIWrapper();
  ~MPIWrapper();
  void Init();
  void AssignLocalRankID();
  void SetGroupNameToMPIGroup(const std::string &group_name, const MPI_Group mpi_group);

  int rank_id_;
  int rank_size_;
  int local_rank_id_;
  MPI_Group world_group_;
  std::map<std::string, MPI_Group> group_name_to_mpi_group_map_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_MPI_WRAPPER_H_
