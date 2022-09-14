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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MPI_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MPI_COMMUNICATION_GROUP_H_

#include <mpi.h>
#include <string>
#include <vector>
#include <memory>
#include "runtime/collective/communication_group.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace device {
namespace cpu {
class MPICommunicationGroup : public CommunicationGroup {
 public:
  explicit MPICommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                 uint32_t global_rank);

  ~MPICommunicationGroup() override = default;

  bool Initialize(void *) override { return true; }
  bool Finalize() override;

  // The OpenMPI groups should be created from the world group.
  bool Initialize(const MPI_Group &world_group);

  const MPI_Comm &mpi_communicator() const { return group_communicator_; }

 private:
  MPI_Group group_;
  MPI_Comm group_communicator_;
};
using MPICommunicationGroupPtr = std::shared_ptr<MPICommunicationGroup>;
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_MPI_COMMUNICATION_GROUP_H_
