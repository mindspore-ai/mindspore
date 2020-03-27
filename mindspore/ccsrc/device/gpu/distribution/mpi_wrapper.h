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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_DISTRIBUTION_MPI_WRAPPER_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_DISTRIBUTION_MPI_WRAPPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <iostream>
#include "device/gpu/distribution/collective_common.h"

namespace mindspore {
namespace device {
namespace gpu {
class MPIWrapper {
 public:
  MPIWrapper(MPIWrapper const &) = delete;
  MPIWrapper &operator=(const MPIWrapper &) = delete;
  static MPIWrapper &instance();
  int local_rank_id() const;

 private:
  MPIWrapper();
  ~MPIWrapper();
  void Init();
  void AssignLocalRankId();

  int rank_id_;
  int rank_size_;
  int local_rank_id_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_DISTRIBUTION_MPI_WRAPPER_H_
