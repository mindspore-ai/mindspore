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

#include "runtime/hardware/gpu/nvidia_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace gpu {
NvidiaCollectiveCommLib::NvidiaCollectiveCommLib() {
  collective_comm_lib_ptr_ = CollectiveInitializer::instance().collective_handle();
}

bool NvidiaCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  MS_LOG(INFO) << "The global rank id of this process is " << global_rank_id_
               << ", global rank size of nccl_world_group is " << global_rank_size_;
  initialized_ = true;
  return true;
}

bool NvidiaCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                       const std::vector<uint32_t> &group_ranks) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(EXCEPTION) << "The NCCL group " << group_name << " has already existed.";
    return false;
  }

  NvidiaCommunicationGroupPtr group =
    std::make_shared<NvidiaCommunicationGroup>(group_name, group_ranks, global_rank_id_);
  MS_EXCEPTION_IF_NULL(group);
  groups_[group_name] = group;
  MS_LOG(INFO) << "NCCL group of " << group_name << " is created. But it's not initialized yet.";
  return true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
