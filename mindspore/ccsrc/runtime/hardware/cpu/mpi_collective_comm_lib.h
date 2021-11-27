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
#include "runtime/hardware/cpu/mpi_communication_group.h"

#ifndef EXPORT_MPI_WRAPPER
#define EXPORT_MPI_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace cpu {
constexpr char kMPIGlobalGroupName[] = "mpi_world_group";
class EXPORT_MPI_WRAPPER MPICollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static MPICollectiveCommLib &GetInstance() {
    static MPICollectiveCommLib instance;
    return instance;
  }

  bool Initialize(uint32_t global_rank = UINT32_MAX, uint32_t global_rank_size = UINT32_MAX) override;

  // Override creating method. Reuse destroying method in base class CollectiveCommunicationLib.
  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 const std::string &group_name, void *stream) override {
    return true;
  }

  bool Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type, uint32_t root_rank,
                 const std::string &group_name, void *stream) override {
    return true;
  }

 private:
  MPICollectiveCommLib();
  ~MPICollectiveCommLib() override = default;

  MPI_Group world_group_;
};
}  // namespace cpu

extern "C" EXPORT_MPI_WRAPPER CollectiveCommunicationLib *communication_lib_instance();
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MPI_COLLECTIVE_COMM_LIB_H_
