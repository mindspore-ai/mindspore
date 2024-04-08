/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/hardware/dummy_ascend_collective_comm_lib.h"
#include <algorithm>
#include <numeric>
#include <memory>
#include "include/common/utils/utils.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/hardware/ascend_communication_group.h"

namespace mindspore {
namespace device {
DummyAscendCollectiveCommLib::DummyAscendCollectiveCommLib() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  global_group_name_ = kHCCLWorldGroup;
}

bool DummyAscendCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  DummyCollectiveCommunicationLib::Initialize(global_rank, global_rank_size, local_rank_id);
  std::string rank_id_str = std::to_string(0);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
  (void)device_context->GetDeprecatedInterface()->OpenTsd(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)hccl::HcclAdapter::GetInstance().InitHccl(device_id, rank_id_str);
  return true;
}

bool DummyAscendCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                            const std::vector<uint32_t> &group_ranks,
                                                            uint32_t local_group_rank, uint32_t local_group_size) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(WARNING) << "The group " << group_name << " has already existed.";
    return true;
  }
  auto group = std::make_shared<ascend::AscendCommunicationGroup>(group_name, group_ranks, GetRankId(group_name),
                                                                  local_group_rank, local_group_size);
  groups_[group_name] = group;
  std::vector<unsigned int> dummy_ranks = {0};
  if (hccl::HcclAdapter::GetInstance().HcclCreateGroup(group_name, UlongToUint(1),
                                                       std::vector<unsigned int>(dummy_ranks).data()) != 0) {
    MS_LOG(ERROR) << "create communicate group" << group_name << "failed";
    return false;
  }

  return true;
}

std::string DummyAscendCollectiveCommLib::HcclInnerCommName(const std::string &group_name) {
  if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    return "";
  }
  CHECK_RET((groups_.size() != 0), true, "The HCCL group " + group_name + " is not initialized.");
  auto group = std::dynamic_pointer_cast<ascend::AscendCommunicationGroup>(groups_.begin()->second);
  CHECK_IF_NULL(group);
  return group->inner_comm_name();
}

bool DummyAscendCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &) {
  std::string group_name = "dummy_group_name";
  bool res = hccl::HcclAdapter::GetInstance().HcclDestroyGroup(group_name);
  if (!res) {
    MS_LOG(ERROR) << "destroy communicate group";
    return false;
  }
  return true;
}

bool DummyAscendCollectiveCommLib::DestroyHcclComm() {
  CHECK_RET((groups_.size() != 0), true, "The HCCL group does not existed.");
  groups_.begin()->second->Finalize();
  groups_.clear();
  bool res = hccl::HcclAdapter::GetInstance().FinalizeHccl();
  if (!res) {
    MS_LOG(ERROR) << "Hccl finalize failed";
    return false;
  }
  return true;
}

}  // namespace device
}  // namespace mindspore
