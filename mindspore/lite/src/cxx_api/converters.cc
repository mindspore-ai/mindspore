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
#include "src/cxx_api/converters.h"
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/context.h"
#include "include/api/context.h"
#include "src/runtime/inner_allocator.h"
#include "src/common/log_adapter.h"

namespace mindspore {
constexpr static int kMaxNumOfDevices = 2;

Status A2L_ConvertContext(Context *a_context, lite::Context *l_context) {
  if ((a_context == nullptr) || (l_context == nullptr)) {
    MS_LOG(ERROR) << "Invalid context pointers.";
    return kLiteNullptr;
  }

  auto device_list = a_context->MutableDeviceInfo();
  if (device_list.size() == 0) {
    MS_LOG(ERROR) << "Invalid device list.";
    return kLiteInputParamInvalid;
  }
  if (device_list.size() > kMaxNumOfDevices) {
    MS_LOG(ERROR) << "Only CPU/CPU & GPU/CPU & NPU mode is supported.";
    return kLiteInputParamInvalid;
  }
  l_context->thread_num_ = a_context->GetThreadNum();
  l_context->enable_parallel_ = a_context->GetEnableParallel();
  l_context->affinity_core_list_ = a_context->GetThreadAffinityCoreList();
  l_context->device_list_.clear();
  if (device_list[0]->GetDeviceType() != kCPU) {
    MS_LOG(ERROR) << "CPU context must be enabled and in the first place of device list.";
    return kLiteInputParamInvalid;
  }

  auto cpu_context = device_list[0]->Cast<CPUDeviceInfo>();
  l_context->allocator = cpu_context->GetAllocator();
  if (l_context->allocator == nullptr) {
    l_context->allocator = Allocator::Create();
    if (l_context->allocator == nullptr) {
      MS_LOG(ERROR) << "Create Allocator failed.";
      return kLiteNullptr;
    }
    MS_LOG(DEBUG) << "Set new allocator.";
    cpu_context->SetAllocator(l_context->allocator);
  }

  if (!IsAffinityModeValid(a_context->GetThreadAffinityMode())) {
    MS_LOG(ERROR)
      << "Invalid affinity mode, only supports 0: no affinities, 1: big cores first, 2: little cores first.";
    return kLiteInputParamInvalid;
  }
  lite::CpuBindMode mode = A2L_ConvertAffinityMode(a_context->GetThreadAffinityMode());

  lite::DeviceInfo cpu_info = {0};
  cpu_info.cpu_device_info_ = {cpu_context->GetEnableFP16(), mode};
  l_context->device_list_.push_back({lite::DT_CPU, cpu_info, cpu_context->GetProvider(),
                                     cpu_context->GetProviderDevice(), cpu_context->GetAllocator()});
  if (device_list.size() == kMaxNumOfDevices) {
    lite::DeviceInfo device_info = {0};
    if (device_list[1]->GetDeviceType() == kGPU) {
      auto gpu_context = device_list[1]->Cast<GPUDeviceInfo>();
      device_info.gpu_device_info_ = {gpu_context->GetEnableFP16()};
      l_context->device_list_.push_back({lite::DT_GPU, device_info, gpu_context->GetProvider(),
                                         gpu_context->GetProviderDevice(), gpu_context->GetAllocator()});
    } else if (device_list[1]->GetDeviceType() == kKirinNPU) {
      auto npu_context = device_list[1]->Cast<KirinNPUDeviceInfo>();
      device_info.npu_device_info_ = {npu_context->GetFrequency()};
      l_context->device_list_.push_back({lite::DT_NPU, device_info});
    } else {
      MS_LOG(ERROR) << "Invalid device.";
      return kLiteInputParamInvalid;
    }
  }
  l_context->delegate = a_context->GetDelegate();
  return kSuccess;
}

Status A2L_ConvertContext(const Context::Data *a_context, lite::Context *l_context) {
  if ((a_context == nullptr) || (l_context == nullptr)) {
    MS_LOG(ERROR) << "Invalid context pointers.";
    return kLiteNullptr;
  }

  auto device_list = a_context->device_info_list;
  if (device_list.size() == 0) {
    MS_LOG(ERROR) << "Invalid device list.";
    return kLiteInputParamInvalid;
  }
  if (device_list.size() > kMaxNumOfDevices) {
    MS_LOG(ERROR) << "Only CPU/CPU & GPU/CPU & NPU mode is supported.";
    return kLiteInputParamInvalid;
  }
  l_context->thread_num_ = a_context->thread_num;
  l_context->enable_parallel_ = a_context->enable_parallel_;
  l_context->affinity_core_list_ = a_context->affinity_core_list_;
  l_context->device_list_.clear();
  if (device_list[0]->GetDeviceType() != kCPU) {
    MS_LOG(ERROR) << "CPU context must be enabled and in the first place of device list.";
    return kLiteInputParamInvalid;
  }

  auto cpu_context = device_list[0]->Cast<CPUDeviceInfo>();
  l_context->allocator = cpu_context->GetAllocator();
  if (l_context->allocator == nullptr) {
    l_context->allocator = Allocator::Create();
    if (l_context->allocator == nullptr) {
      MS_LOG(ERROR) << "Create Allocator failed.";
      return kLiteNullptr;
    }
    MS_LOG(DEBUG) << "Set new allocator.";
    cpu_context->SetAllocator(l_context->allocator);
  }

  if (!IsAffinityModeValid(a_context->affinity_mode_)) {
    MS_LOG(ERROR)
      << "Invalid affinity mode, only supports 0: no affinities, 1: big cores first, 2: little cores first.";
    return kLiteInputParamInvalid;
  }
  lite::CpuBindMode mode = A2L_ConvertAffinityMode(a_context->affinity_mode_);

  lite::DeviceInfo cpu_info = {0};
  cpu_info.cpu_device_info_ = {cpu_context->GetEnableFP16(), mode};
  l_context->device_list_.push_back({lite::DT_CPU, cpu_info, cpu_context->GetProvider(),
                                     cpu_context->GetProviderDevice(), cpu_context->GetAllocator()});
  if (device_list.size() == kMaxNumOfDevices) {
    lite::DeviceInfo device_info = {0};
    if (device_list[1]->GetDeviceType() == kGPU) {
      auto gpu_context = device_list[1]->Cast<GPUDeviceInfo>();
      device_info.gpu_device_info_ = {gpu_context->GetEnableFP16()};
      l_context->device_list_.push_back({lite::DT_GPU, device_info, gpu_context->GetProvider(),
                                         gpu_context->GetProviderDevice(), gpu_context->GetAllocator()});
    } else if (device_list[1]->GetDeviceType() == kKirinNPU) {
      auto npu_context = device_list[1]->Cast<KirinNPUDeviceInfo>();
      device_info.npu_device_info_ = {npu_context->GetFrequency()};
      l_context->device_list_.push_back({lite::DT_NPU, device_info});
    } else {
      MS_LOG(ERROR) << "Invalid device.";
      return kLiteInputParamInvalid;
    }
  }
  l_context->delegate = a_context->delegate;
  return kSuccess;
}
}  // namespace mindspore
