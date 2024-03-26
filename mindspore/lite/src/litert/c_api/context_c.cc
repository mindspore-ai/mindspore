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

#include "src/litert/c_api/context_c.h"
#include "include/c_api/context_c.h"
#include "include/api/context.h"
#include "src/common/log_adapter.h"

// ================ Context ================
MSContextHandle MSContextCreate() {
  auto impl = new (std::nothrow) mindspore::Context();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  return static_cast<MSContextHandle>(impl);
}

void MSContextDestroy(MSContextHandle *context) {
  if (context != nullptr && *context != nullptr) {
    auto impl = static_cast<mindspore::Context *>(*context);
    delete impl;
    *context = nullptr;
  }
}

void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  impl->SetThreadNum(thread_num);
}

int32_t MSContextGetThreadNum(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  return impl->GetThreadNum();
}

void MSContextSetThreadAffinityMode(MSContextHandle context, int mode) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  impl->SetThreadAffinity(mode);
  return;
}

int MSContextGetThreadAffinityMode(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return -1;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  return impl->GetThreadAffinityMode();
}

void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t *core_list, size_t core_num) {
  if (context == nullptr || core_list == nullptr) {
    MS_LOG(ERROR) << "context or core_list is nullptr.";
    return;
  }
  const std::vector<int32_t> vec_core_list(core_list, core_list + core_num);
  auto impl = static_cast<mindspore::Context *>(context);
  impl->SetThreadAffinity(vec_core_list);
  return;
}

const int32_t *MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t *core_num) {
  if (context == nullptr || core_num == nullptr) {
    MS_LOG(ERROR) << "context or core_num is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  auto affinity_core_list = impl->GetThreadAffinityCoreList();
  *core_num = affinity_core_list.size();
  int32_t *core_list = static_cast<int32_t *>(malloc((*core_num) * sizeof(int32_t)));
  if (core_list == nullptr) {
    MS_LOG(ERROR) << "malloc core_list is null.";
    return nullptr;
  }
  for (size_t i = 0; i < affinity_core_list.size(); i++) {
    core_list[i] = affinity_core_list[i];
  }
  return core_list;
}

void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  impl->SetEnableParallel(is_parallel);
}

bool MSContextGetEnableParallel(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  return impl->GetEnableParallel();
}

void MSContextAddDeviceInfo(MSContextHandle context, MSDeviceInfoHandle device_info) {
  if (context == nullptr || device_info == nullptr) {
    MS_LOG(ERROR) << "context or device_info is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::Context *>(context);
  std::shared_ptr<mindspore::DeviceInfoContext> device(static_cast<mindspore::DeviceInfoContext *>(device_info));
  impl->MutableDeviceInfo().push_back(device);
}

// ================ DeviceInfo ================
MSDeviceInfoHandle MSDeviceInfoCreate(MSDeviceType device_type) {
  mindspore::DeviceInfoContext *impl;
  if (kMSDeviceTypeCPU == device_type) {
    impl = new (std::nothrow) mindspore::CPUDeviceInfo();
  } else if (kMSDeviceTypeGPU == device_type) {
    impl = new (std::nothrow) mindspore::GPUDeviceInfo();
  } else if (kMSDeviceTypeKirinNPU == device_type) {
    impl = new (std::nothrow) mindspore::KirinNPUDeviceInfo();
  } else {
    MS_LOG(ERROR) << "device_type is invalid.";
    impl = nullptr;
  }
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  return static_cast<MSDeviceInfoHandle>(impl);
}

void MSDeviceInfoDestroy(MSDeviceInfoHandle *device_info) {
  if (device_info != nullptr && *device_info != nullptr) {
    auto impl = static_cast<mindspore::DeviceInfoContext *>(*device_info);
    delete impl;
    *device_info = nullptr;
  }
}

void MSDeviceInfoSetProvider(MSDeviceInfoHandle device_info, const char *provider) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  impl->SetProvider(provider);
}

const char *MSDeviceInfoGetProvider(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  char *provider = static_cast<char *>(malloc(impl->GetProvider().size() + 1));
  if (provider == nullptr) {
    MS_LOG(ERROR) << "malloc provider is null.";
    return nullptr;
  }
  for (size_t i = 0; i < impl->GetProvider().size(); i++) {
    provider[i] = impl->GetProvider()[i];
  }
  provider[impl->GetProvider().size()] = '\0';
  return provider;
}

void MSDeviceInfoSetProviderDevice(MSDeviceInfoHandle device_info, const char *device) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  impl->SetProviderDevice(device);
}

const char *MSDeviceInfoGetProviderDevice(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  char *provider_device = static_cast<char *>(malloc(impl->GetProviderDevice().size() + 1));
  if (provider_device == nullptr) {
    MS_LOG(ERROR) << "malloc provider_device is null.";
    return nullptr;
  }
  for (size_t i = 0; i < impl->GetProviderDevice().size(); i++) {
    provider_device[i] = impl->GetProviderDevice()[i];
  }
  provider_device[impl->GetProviderDevice().size()] = '\0';
  return provider_device;
}

MSDeviceType MSDeviceInfoGetDeviceType(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return kMSDeviceTypeInvalid;
  }
  auto impl = static_cast<mindspore::DeviceInfoContext *>(device_info);
  return static_cast<MSDeviceType>(impl->GetDeviceType());
}

void MSDeviceInfoSetEnableFP16(MSDeviceInfoHandle device_info, bool is_fp16) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (kMSDeviceTypeCPU == static_cast<MSDeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::CPUDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else if (kMSDeviceTypeGPU == static_cast<MSDeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::GPUDeviceInfo *>(device_info);
    impl->SetEnableFP16(is_fp16);
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

bool MSDeviceInfoGetEnableFP16(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return false;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (kMSDeviceTypeCPU == static_cast<MSDeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::CPUDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else if (kMSDeviceTypeGPU == static_cast<MSDeviceType>(impl_device->GetDeviceType())) {
    auto impl = static_cast<mindspore::GPUDeviceInfo *>(device_info);
    return impl->GetEnableFP16();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature. device_type: " << impl_device->GetDeviceType();
    return false;
  }
}

void MSDeviceInfoSetFrequency(MSDeviceInfoHandle device_info, int frequency) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (static_cast<MSDeviceType>(impl_device->GetDeviceType()) == kMSDeviceTypeKirinNPU) {
    auto impl = static_cast<mindspore::KirinNPUDeviceInfo *>(device_info);
    impl->SetFrequency(frequency);
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

int MSDeviceInfoGetFrequency(const MSDeviceInfoHandle device_info) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return -1;
  }
  auto impl_device = static_cast<mindspore::DeviceInfoContext *>(device_info);
  if (static_cast<MSDeviceType>(impl_device->GetDeviceType()) == kMSDeviceTypeKirinNPU) {
    auto impl = static_cast<mindspore::KirinNPUDeviceInfo *>(device_info);
    return impl->GetFrequency();
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
    return -1;
  }
}
