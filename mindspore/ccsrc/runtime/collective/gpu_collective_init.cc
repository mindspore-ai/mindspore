/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/collective/gpu_collective_init.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
void GPUCollectiveInitializer::InitCollective() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kGPUDevice) {
    MS_LOG(EXCEPTION) << "You are trying to call 'init('nccl')', Please check "
                         "this MindSpore package is GPU version and built with NCCL.";
  }
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kGPUDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto deprecated_ptr = device_context->GetDeprecatedInterface();
  MS_EXCEPTION_IF_NULL(deprecated_ptr);
  deprecated_ptr->GPUInitCollective();
}

void GPUCollectiveInitializer::FinalizeCollective() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kGPUDevice) {
    MS_LOG(EXCEPTION) << "You are trying to call 'finalize('nccl')', Please check "
                         "this MindSpore package is GPU version and built with NCCL.";
  }
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kGPUDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto deprecated_ptr = device_context->GetDeprecatedInterface();
  MS_EXCEPTION_IF_NULL(deprecated_ptr);
  deprecated_ptr->GPUFinalizeCollective();
}

uint32_t GPUCollectiveInitializer::GetRankID(const std::string &group_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kGPUDevice) {
    MS_LOG(EXCEPTION) << "You are trying to call 'GetRankID', Please check "
                         "this MindSpore package is GPU version and built with NCCL.";
  }
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kGPUDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto deprecated_ptr = device_context->GetDeprecatedInterface();
  MS_EXCEPTION_IF_NULL(deprecated_ptr);
  return deprecated_ptr->GPUGetRankID(group_name);
}

uint32_t GPUCollectiveInitializer::GetRankSize(const std::string &group_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kGPUDevice) {
    MS_LOG(EXCEPTION) << "You are trying to call 'GetRankSize', Please check "
                         "this MindSpore package is GPU version and built with NCCL.";
  }
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kGPUDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto deprecated_ptr = device_context->GetDeprecatedInterface();
  MS_EXCEPTION_IF_NULL(deprecated_ptr);
  return deprecated_ptr->GPUGetRankSize(group_name);
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
