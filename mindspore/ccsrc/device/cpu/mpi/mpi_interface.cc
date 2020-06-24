/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <pybind11/operators.h>
#include "device/cpu/mpi/mpi_adapter.h"

namespace mindspore {
namespace device {
namespace cpu {
int get_rank_id() { return MPIAdapter::Instance()->GetRankId(); }

int get_rank_size() { return MPIAdapter::Instance()->GetRankSize(); }

PYBIND11_MODULE(_ms_mpi, mpi_interface) {
  mpi_interface.doc() = "mindspore mpi python wrapper";
  mpi_interface.def("get_rank_id", &get_rank_id, "get rank id");
  mpi_interface.def("get_rank_size", &get_rank_size, "get rank size");
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
