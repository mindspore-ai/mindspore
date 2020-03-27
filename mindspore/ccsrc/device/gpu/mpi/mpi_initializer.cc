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

#include "device/gpu/mpi/mpi_initializer.h"

#include <mpi.h>
#include <pybind11/operators.h>
#include <iostream>

namespace mindspore {
namespace device {
namespace gpu {
MPIInitializer::MPIInitializer() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id_);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_size_);
}

MPIInitializer &MPIInitializer::GetInstance() {
  static MPIInitializer instance;
  return instance;
}

int MPIInitializer::get_rank_id() { return MPIInitializer::GetInstance().rank_id_; }

int MPIInitializer::get_rank_size() { return MPIInitializer::GetInstance().rank_size_; }

PYBIND11_MODULE(_ms_mpi, mpi_initializer) {
  mpi_initializer.doc() = "mindspore mpi python wrapper";
  mpi_initializer.def("get_rank_id", &MPIInitializer::get_rank_id, "get rank id");
  mpi_initializer.def("get_rank_size", &MPIInitializer::get_rank_size, "get rank size");
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
