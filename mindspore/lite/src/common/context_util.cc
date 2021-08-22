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

#include "src/common/context_util.h"
#include <set>
#include <map>
#include <memory>
#include <string>
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
namespace {
template <class T>
void PassBasicProperties(std::shared_ptr<T> device_info, const lite::DeviceContext &device_context) {
  device_info->SetProvider(device_context.provider_);
  device_info->SetProviderDevice(device_context.provider_device_);
  device_info->SetAllocator(device_context.allocator_);
}

std::shared_ptr<mindspore::CPUDeviceInfo> CPUDeviceInfoFromCPUDeviceContext(const lite::DeviceContext &cpu_context) {
  if (cpu_context.device_type_ != DT_CPU) {
    MS_LOG(ERROR) << "function input parameter is not cpu context.";
    return nullptr;
  }
  auto cpu_info = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_info->SetEnableFP16(cpu_context.device_info_.cpu_device_info_.enable_float16_);
  PassBasicProperties(cpu_info, cpu_context);
  return cpu_info;
}

std::shared_ptr<mindspore::GPUDeviceInfo> GPUDeviceInfoFromGPUDeviceContext(const lite::DeviceContext &gpu_context) {
  if (gpu_context.device_type_ != DT_GPU) {
    MS_LOG(ERROR) << "function input parameter is not gpu context.";
    return nullptr;
  }
  auto gpu_info = std::make_shared<mindspore::GPUDeviceInfo>();
  gpu_info->SetEnableFP16(gpu_context.device_info_.gpu_device_info_.enable_float16_);
  PassBasicProperties(gpu_info, gpu_context);
  return gpu_info;
}

std::shared_ptr<mindspore::KirinNPUDeviceInfo> NPUDeviceInfoFromNPUDeviceContext(
  const lite::DeviceContext &npu_context) {
  if (npu_context.device_type_ != DT_NPU) {
    MS_LOG(ERROR) << "function input parameter is not npu context.";
    return nullptr;
  }
  auto npu_info = std::make_shared<mindspore::KirinNPUDeviceInfo>();
  npu_info->SetFrequency(npu_context.device_info_.npu_device_info_.frequency_);
  PassBasicProperties(npu_info, npu_context);
  return npu_info;
}
}  // namespace

mindspore::Context *MSContextFromContext(const lite::Context *context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    return nullptr;
  }
  auto ms_context = new (std::nothrow) mindspore::Context();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "New Context failed";
    return nullptr;
  }
  ms_context->SetThreadNum(context->thread_num_);
  ms_context->SetThreadAffinity(context->affinity_core_list_);
  ms_context->SetEnableParallel(context->enable_parallel_);
  ms_context->SetDelegate(context->delegate);
  auto &device_infos = ms_context->MutableDeviceInfo();
  std::map<DeviceType, std::function<std::shared_ptr<mindspore::DeviceInfoContext>(const lite::DeviceContext &)>>
    transfer_funcs = {{DT_CPU, CPUDeviceInfoFromCPUDeviceContext},
                      {DT_GPU, GPUDeviceInfoFromGPUDeviceContext},
                      {DT_NPU, NPUDeviceInfoFromNPUDeviceContext}};
  for (auto &device_context : context->device_list_) {
    auto device_type = device_context.device_type_;
    if (transfer_funcs.find(device_type) == transfer_funcs.end()) {
      MS_LOG(ERROR) << "device type is invalid.";
      return nullptr;
    }
    auto device_info = transfer_funcs[device_type](device_context);
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "transfer device context to device info failed.";
      return nullptr;
    }
    if (device_type == DT_CPU) {
      ms_context->SetThreadAffinity(device_context.device_info_.cpu_device_info_.cpu_bind_mode_);
    }
    device_infos.push_back(device_info);
  }
  return ms_context;
}
}  // namespace lite
}  // namespace mindspore
