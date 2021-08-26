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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_MPI_PYCC_H
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_MPI_PYCC_H

#include <string>
#include <vector>
#include "runtime/device/ascend/distribute/collective_group_wrapper.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
class MpiPycc {
 public:
  MpiPycc(MpiPycc const &) = delete;
  MpiPycc &operator=(const MpiPycc &) = delete;
  static MpiPycc &instance();
  static int GetDeviceID();
  static int GetRankId(const std::string &group);
  static int GetRankSize(const std::string &group);
  static void CreateGroup(const std::string &group, const std::vector<unsigned int> &ranks);

 private:
  MpiPycc() = default;
  ~MpiPycc() = default;
};
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_MPI_PYCC_H
