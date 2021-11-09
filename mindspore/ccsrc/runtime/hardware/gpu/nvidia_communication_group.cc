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

#include "runtime/hardware/gpu/nvidia_communication_group.h"

namespace mindspore {
namespace device {
namespace gpu {
NvidiaCommunicationGroup::NvidiaCommunicationGroup(const std::string name, const std::vector<uint32_t> &group_ranks,
                                                   uint32_t global_rank)
    : CommunicationGroup(name, group_ranks, global_rank) {
  collective_comm_lib_ptr_ = CollectiveInitializer::instance().collective_handle();
}

bool NvidiaCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return false;
  }

  // The unique id is broadcasted by the root rank.
  unique_id_ = *(static_cast<ncclUniqueId *>(root_info));

  // Initialize the NCCL communicator while the group created. Pay attention that 'ncclCommInitRank' should be called
  // after GPU device id is set.
  MS_EXCEPTION_IF_NULL(collective_comm_lib_ptr_);
  auto comm_init_rank =
    reinterpret_cast<NCCLCommInitRank>(dlsym(const_cast<void *>(collective_comm_lib_ptr_), "NCCLCommInitRank"));
  MS_EXCEPTION_IF_NULL(comm_init_rank);

  MS_LOG(INFO) << "Start initializing NCCL communicator for group " << name_;
  uint32_t group_rank = GetGroupRank(global_rank_);
  CHECK_NCCL_RET((*comm_init_rank)(&comm_, SizeToInt(size_), unique_id_, UintToInt(group_rank)),
                 "Initializing NCCL communicator failed.");
  MS_LOG(INFO) << "NCCL communicator for group " << name_ << " is successfully initialized.";

  initialized_ = true;
  return true;
}

bool NvidiaCommunicationGroup::Finalize() {
  if (!initialized_) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(collective_comm_lib_ptr_);
  auto comm_abort =
    reinterpret_cast<NCCLCommAbort>(dlsym(const_cast<void *>(collective_comm_lib_ptr_), "NCCLCommAbort"));
  MS_EXCEPTION_IF_NULL(comm_abort);
  auto comm_destroy =
    reinterpret_cast<NCCLCommDestroy>(dlsym(const_cast<void *>(collective_comm_lib_ptr_), "NCCLCommDestroy"));
  MS_EXCEPTION_IF_NULL(comm_destroy);

  CHECK_NCCL_RET((*comm_abort)(comm_), "Failed to abort NCCL communicator.");
  CHECK_NCCL_RET((*comm_destroy)(comm_), "Failed to destroy NCCL communicator.");
  initialized_ = false;
  return true;
}

void *NvidiaCommunicationGroup::GenerateRootInfo() {
  MS_EXCEPTION_IF_NULL(collective_comm_lib_ptr_);
  auto nccl_id_funcptr =
    reinterpret_cast<NcclUniqueId>(dlsym(const_cast<void *>(collective_comm_lib_ptr_), "nccl_unique_id"));
  MS_EXCEPTION_IF_NULL(nccl_id_funcptr);
  unique_id_ = (*nccl_id_funcptr)();
  return &unique_id_;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
