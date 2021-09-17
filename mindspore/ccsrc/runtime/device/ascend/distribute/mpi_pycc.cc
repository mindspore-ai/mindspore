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

#include "runtime/device/ascend/distribute/mpi_pycc.h"
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <vector>

namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
MpiPycc &MpiPycc::instance() {
  static MpiPycc instance = {};
  return instance;
}

int MpiPycc::GetDeviceID() { return GetDeviceId(); }
int MpiPycc::GetRankId(const std::string &group) { return GetRankIdByGroup(group); }
int MpiPycc::GetRankSize(const std::string &group) { return GetGroupSize(group); }
void MpiPycc::CreateGroup(const std::string &group, const std::vector<unsigned int> &ranks) {
  CreateCommForGroup(group, ranks);
}

// cppcheck-suppress syntaxError
PYBIND11_MODULE(_ascend_mpi, mpi_initializer) {
  mpi_initializer.def("get_device_id", &MpiPycc::GetDeviceID, "get device id");
  mpi_initializer.def("get_rank_id", &MpiPycc::GetRankId, "get rank id");
  mpi_initializer.def("get_rank_size", &MpiPycc::GetRankSize, "get rank size");
  mpi_initializer.def("create_group", &MpiPycc::CreateGroup, "create group");
}
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
