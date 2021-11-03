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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MPI_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MPI_COLLECTIVE_COMM_LIB_H_

#include <memory>
#include <vector>
#include <string>
#include "runtime/hardware/collective/collective_communication_lib.h"

namespace mindspore {
namespace device {
namespace cpu {
class MPICollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static MPICollectiveCommLib &GetInstance() {
    static MPICollectiveCommLib instance;
    return instance;
  }

  void Initialize() override;
  void Finalize() override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;
  bool DestroyCommunicationGroup(const std::string &group_name) override;

 private:
  MPICollectiveCommLib() {}
  ~MPICollectiveCommLib() override;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MPI_COLLECTIVE_COMM_LIB_H_
