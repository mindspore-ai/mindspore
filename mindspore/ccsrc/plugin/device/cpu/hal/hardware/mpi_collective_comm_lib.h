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

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "runtime/collective/collective_communication_lib.h"
#include "plugin/device/cpu/hal/hardware/mpi_communication_group.h"

#ifndef EXPORT_MPI_WRAPPER
#define EXPORT_MPI_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace cpu {
// Map of collective operation data type to MPI data type.
const std::map<TypeId, MPI_Datatype> kMPIDataTypeMap = {{TypeId::kNumberTypeInt8, MPI_BYTE},
                                                        {TypeId::kNumberTypeUInt8, MPI_UNSIGNED_CHAR},
                                                        {TypeId::kNumberTypeInt32, MPI_INT},
                                                        {TypeId::kNumberTypeInt, MPI_INT},
                                                        {TypeId::kNumberTypeUInt32, MPI_UNSIGNED},
                                                        {TypeId::kNumberTypeInt64, MPI_LONG_LONG},
                                                        {TypeId::kNumberTypeUInt64, MPI_UNSIGNED_LONG_LONG},
                                                        {TypeId::kNumberTypeFloat32, MPI_FLOAT},
                                                        {TypeId::kNumberTypeFloat, MPI_FLOAT},
                                                        {TypeId::kNumberTypeFloat64, MPI_DOUBLE}};

// Map of reduce type to MPI reduce type.
const std::map<CollectiveOpReduceType, MPI_Op> kMPIReduceTypeMap = {{CollectiveOpReduceType::Reduce_Sum, MPI_SUM},
                                                                    {CollectiveOpReduceType::Reduce_Prod, MPI_PROD},
                                                                    {CollectiveOpReduceType::Reduce_Min, MPI_MIN},
                                                                    {CollectiveOpReduceType::Reduce_Max, MPI_MAX}};

constexpr char kMPIGlobalGroupName[] = "mpi_world_group";
class EXPORT_MPI_WRAPPER MPICollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static MPICollectiveCommLib &GetInstance() {
    static MPICollectiveCommLib instance;
    return instance;
  }

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool AllGatherHostHashName(size_t host_hash_name, std::vector<size_t> *host_hash_names) const override;

  bool BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) override;

  // Override creating method. Reuse destroying method in base class CollectiveCommunicationLib.
  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size) override;

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
