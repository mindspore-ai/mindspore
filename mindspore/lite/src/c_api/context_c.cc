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
#include "include/c_api/context_c.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/delegate.h"
#include "include/api/allocator.h"
#include "src/cxx_api/context.h"
#include "src/common/log_adapter.h"

// ================ Context ================
MSContextHandle MSContextCreate() {
  auto impl = new (std::nothrow) mindspore::Context::Data;
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  return static_cast<MSContextHandle>(impl);
}

void MSContextDestroy(MSContextHandle context) {
  if (context != nullptr) {
    auto impl = static_cast<mindspore::Context::Data *>(context);
    delete impl;
  }
}

void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  impl->thread_num = thread_num;
}

int32_t MSContextGetThreadNum(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  return impl->thread_num;
}

void MSContextSetThreadAffinityMode(MSContextHandle context, int mode) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  impl->affinity_mode_ = mode;
  return;
}

int MSContextGetThreadAffinityMode(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  return impl->affinity_mode_;
}

void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t *core_list, size_t core_num) {
  if (context == nullptr || core_list == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  const std::vector<int32_t> vec_core_list(core_list, core_list + core_num);
  auto impl = static_cast<mindspore::Context::Data *>(context);
  impl->affinity_core_list_ = vec_core_list;
  return;
}

const int32_t *MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t *core_num) {
  if (context == nullptr || core_num == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  *core_num = impl->affinity_core_list_.size();
  return impl->affinity_core_list_.data();
}

void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  impl->enable_parallel_ = is_parallel;
}

bool MSContextGetEnableParallel(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  return impl->enable_parallel_;
}

void MSContextAddDeviceInfo(MSContextHandle context, MSDeviceInfoHandle device_info) {
  if (context == nullptr || device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context::Data *>(context);
  std::shared_ptr<mindspore::DeviceInfoContext> device(static_cast<mindspore::DeviceInfoContext *>(device_info));
  impl->device_info_list.push_back(device);
}

// ================ DeviceInfo ================
MSDeviceInfoHandle MSDeviceInfoCreate(MSDeviceType device_type) {
  mindspore::DeviceInfoContext *impl = nullptr;
  if (device_type == kMSDeviceTypeCPU) {
    impl = new (std::nothrow) mindspore::CPUDeviceInfo();
  } else if (device_type == kMSDeviceTypeGPU) {
    impl = new (std::nothrow) mindspore::GPUDeviceInfo();
  } else if (device_type == kMSDeviceTypeKirinNPU) {
    impl = new (std::nothrow) mindspore::KirinNPUDeviceInfo();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature. device_type: " << device_type;
    return nullptr;
  }
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  return static_cast<MSDeviceInfoHandle>(impl);
}

void MSDeviceInfoDestroy(MSDeviceInfoHandle device_info) {
  if (device_info != nullptr) {
    auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
    delete impl;
  }
}

void MSDeviceInfoSetProvider(MSDeviceInfoHandle device_info, const char *provider) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  return impl->SetProvider(provider);
}

const char *MSDeviceInfoGetProvider(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  return impl->GetProvider().c_str();
}

void MSDeviceInfoSetProviderDevice(MSDeviceInfoHandle device_info, const char *device) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  return impl->SetProviderDevice(device);
}

const char *MSDeviceInfoGetProviderDevice(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  return impl->GetProviderDevice().c_str();
}

MSDeviceType MSDeviceInfoGetDeviceType(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSDeviceTypeInvalid;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  auto device_type = impl->GetDeviceType();
  return static_cast<MSDeviceType>(device_type);
}

void MSDeviceInfoSetEnableFP16(MSDeviceInfoHandle device_info, bool is_fp16) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto device_type = static_cast<mindspore::DeviceInfoContext *>(device_info)->GetDeviceType();
  if (static_cast<MSDeviceType>(device_type) == kMSDeviceTypeCPU) {
    auto impl = static_cast<mindspore::CPUDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else if (static_cast<MSDeviceType>(device_type) == kMSDeviceTypeGPU) {
    auto impl = static_cast<mindspore::GPUDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

bool MSDeviceInfoGetEnableFP16(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return false;
  }
  auto device_type = static_cast<mindspore::DeviceInfoContext *>(device_info)->GetDeviceType();
  if (static_cast<MSDeviceType>(device_type) == kMSDeviceTypeCPU) {
    auto impl = static_cast<mindspore::CPUDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else if (static_cast<MSDeviceType>(device_type) == kMSDeviceTypeGPU) {
    auto impl = static_cast<mindspore::GPUDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature. device_type: " << device_type;
    return false;
  }
}

void MSDeviceInfoSetFrequency(MSDeviceInfoHandle device_info, int frequency) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto device_type = static_cast<mindspore::DeviceInfoContext *>(device_info)->GetDeviceType();
  if (static_cast<MSDeviceType>(device_type) == kMSDeviceTypeKirinNPU) {
    auto impl = static_cast<mindspore::KirinNPUDeviceInfo *>(device_info);
    impl->SetFrequency(frequency);
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

int MSDeviceInfoGetFrequency(const MSDeviceInfoHandle device_info) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return -1;
  }
  auto device_type = static_cast<mindspore::DeviceInfoContext *>(device_info)->GetDeviceType();
  if (static_cast<MSDeviceType>(device_type) == kMSDeviceTypeKirinNPU) {
    auto impl = static_cast<mindspore::KirinNPUDeviceInfo *>(device_info);
    return impl->GetFrequency();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
    return -1;
  }
}
