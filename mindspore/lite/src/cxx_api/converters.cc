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
#include "nnacl/op_base.h"

namespace mindspore {
constexpr static int kMaxNumOfDevices = 3;

Status AddCpuDevice(Context *a_context, lite::InnerContext *l_context, DeviceInfoContext *device) {
  auto cpu_context = device->Cast<CPUDeviceInfo>();
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
  return kSuccess;
}

Status AddGpuDevice(Context *a_context, lite::InnerContext *l_context, DeviceInfoContext *device) {
  lite::DeviceInfo device_info = {0};
  auto gpu_context = device->Cast<GPUDeviceInfo>();
  device_info.gpu_device_info_ = {gpu_context->GetEnableFP16(), gpu_context->GetDeviceID()};
  l_context->device_list_.push_back({lite::DT_GPU, device_info, gpu_context->GetProvider(),
                                     gpu_context->GetProviderDevice(), gpu_context->GetAllocator()});
  return kSuccess;
}

Status AddNpuDevice(Context *a_context, lite::InnerContext *l_context, DeviceInfoContext *device) {
  lite::DeviceInfo device_info = {0};
  auto npu_context = device->Cast<KirinNPUDeviceInfo>();
  device_info.npu_device_info_ = {npu_context->GetFrequency()};
  l_context->device_list_.push_back({lite::DT_NPU, device_info});
  return kSuccess;
}

Status AddAscend310Device(Context *a_context, lite::InnerContext *l_context, DeviceInfoContext *device) {
  lite::DeviceInfo device_info = {0};
  auto ascend310_context = device->Cast<Ascend310DeviceInfo>();
  device_info.ascend310_device_info_ = {ascend310_context->GetDeviceID(), ascend310_context->GetDynamicBatchSize()};
  l_context->device_list_.push_back({lite::DT_ASCEND310, device_info});
  return kSuccess;
}

Status A2L_ConvertContext(Context *a_context, lite::InnerContext *l_context) {
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
    MS_LOG(ERROR) << "Device support Max: " << kMaxNumOfDevices;
    return kLiteInputParamInvalid;
  }
  l_context->thread_num_ = a_context->GetThreadNum();
  l_context->enable_parallel_ = a_context->GetEnableParallel();
  l_context->affinity_core_list_ = a_context->GetThreadAffinityCoreList();
  l_context->device_list_.clear();

  Status error_code;
  for (auto &device : device_list) {
    MS_CHECK_TRUE_RET(device != nullptr, kLiteNullptr);
    if (device->GetDeviceType() == kCPU) {
      error_code = AddCpuDevice(a_context, l_context, device.get());
    } else if (device->GetDeviceType() == kGPU) {
      error_code = AddGpuDevice(a_context, l_context, device.get());
    } else if (device->GetDeviceType() == kKirinNPU) {
      error_code = AddNpuDevice(a_context, l_context, device.get());
    } else if (device->GetDeviceType() == kAscend310) {
      error_code = AddAscend310Device(a_context, l_context, device.get());
    } else {
      MS_LOG(ERROR) << "Invalid device.";
      return kLiteInputParamInvalid;
    }

    if (error_code != kSuccess) {
      MS_LOG(ERROR) << "Add device failed!";
      return error_code;
    }
  }

  l_context->delegate = a_context->GetDelegate();
  return kSuccess;
}
}  // namespace mindspore
