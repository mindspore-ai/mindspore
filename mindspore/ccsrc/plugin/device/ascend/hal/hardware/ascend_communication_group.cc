/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_communication_group.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendCommunicationGroup::AscendCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                                   uint32_t global_rank, uint32_t local_group_rank,
                                                   uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      unique_id_({}),
      comm_(nullptr) {}

bool AscendCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return false;
  }
  unique_id_ = *(static_cast<HcclRootInfo *>(root_info));
  uint32_t group_rank = GetGroupRank(global_rank_);
  if (HcclCommInitRootInfo(static_cast<uint32_t>(size_), &unique_id_, static_cast<uint32_t>(group_rank), &comm_) !=
      static_cast<int32_t>(HCCL_SUCCESS)) {
    const string &error_message = ErrorManager::GetInstance().GetErrorMessage();
    MS_LOG(ERROR) << "HcclCommInitRootInfo failed. " + error_message;
  }
  initialized_ = true;
  return true;
}

bool AscendCommunicationGroup::Finalize() {
  if (!initialized_) {
    return false;
  }
  RETURN_IF_FALSE_WITH_LOG(HcclCommDestroy(comm_) == static_cast<int32_t>(HCCL_SUCCESS),
                           "Failed to destroy HCCL communicator.");
  initialized_ = false;
  comm_ = nullptr;
  return true;
}

void *AscendCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  *root_info_size = sizeof(unique_id_);
  uint32_t group_rank = GetGroupRank(global_rank_);
  if (group_rank == 0) {
    CHECK_RET(HcclGetRootInfo(&unique_id_), static_cast<int32_t>(HCCL_SUCCESS), "Failed to get HCCL unique id.");
  }
  return &unique_id_;
}

const HcclComm &AscendCommunicationGroup::hccl_communicator() const { return comm_; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
