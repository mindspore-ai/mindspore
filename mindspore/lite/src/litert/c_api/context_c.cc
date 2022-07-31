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
#include "src/litert/c_api/context_c.h"
#include "src/common/log_adapter.h"

// ================ Context ================
MSContextHandle MSContextCreate() {
  auto impl = new (std::nothrow) mindspore::ContextC;
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  return static_cast<MSContextHandle>(impl);
}

void MSContextDestroy(MSContextHandle *context) {
  if (context != nullptr && *context != nullptr) {
    auto impl = static_cast<mindspore::ContextC *>(*context);
    delete impl;
    *context = nullptr;
  }
}

void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->thread_num = thread_num;
}

int32_t MSContextGetThreadNum(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  return impl->thread_num;
}

void MSContextSetThreadAffinityMode(MSContextHandle context, int mode) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->affinity_mode = mode;
  return;
}

int MSContextGetThreadAffinityMode(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return -1;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  return impl->affinity_mode;
}

void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t *core_list, size_t core_num) {
  if (context == nullptr || core_list == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  const std::vector<int32_t> vec_core_list(core_list, core_list + core_num);
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->affinity_core_list = vec_core_list;
  return;
}

const int32_t *MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t *core_num) {
  if (context == nullptr || core_num == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  *core_num = impl->affinity_core_list.size();
  return impl->affinity_core_list.data();
}

void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  impl->enable_parallel = is_parallel;
}

bool MSContextGetEnableParallel(const MSContextHandle context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  return impl->enable_parallel;
}

void MSContextAddDeviceInfo(MSContextHandle context, MSDeviceInfoHandle device_info) {
  if (context == nullptr || device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::ContextC *>(context);
  std::shared_ptr<mindspore::DeviceInfoC> device(static_cast<mindspore::DeviceInfoC *>(device_info));
  impl->device_info_list.push_back(device);
}

// ================ DeviceInfo ================
MSDeviceInfoHandle MSDeviceInfoCreate(MSDeviceType device_type) {
  mindspore::DeviceInfoC *impl = new (std::nothrow) mindspore::DeviceInfoC;
  if (impl == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed.";
    return nullptr;
  }
  impl->device_type = device_type;
  return static_cast<MSDeviceInfoHandle>(impl);
}

void MSDeviceInfoDestroy(MSDeviceInfoHandle *device_info) {
  if (device_info != nullptr && *device_info != nullptr) {
    auto impl = static_cast<mindspore::DeviceInfoC *>(*device_info);
    delete impl;
    *device_info = nullptr;
  }
}

void MSDeviceInfoSetProvider(MSDeviceInfoHandle device_info, const char *provider) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  impl->provider = provider;
}

const char *MSDeviceInfoGetProvider(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  return impl->provider.c_str();
}

void MSDeviceInfoSetProviderDevice(MSDeviceInfoHandle device_info, const char *device) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  impl->provider_device = device;
}

const char *MSDeviceInfoGetProviderDevice(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  return impl->provider_device.c_str();
}

MSDeviceType MSDeviceInfoGetDeviceType(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSDeviceTypeInvalid;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  return impl->device_type;
}

void MSDeviceInfoSetEnableFP16(MSDeviceInfoHandle device_info, bool is_fp16) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  if (impl->device_type == kMSDeviceTypeCPU || impl->device_type == kMSDeviceTypeGPU) {
    impl->enable_fp16 = is_fp16;
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

bool MSDeviceInfoGetEnableFP16(const MSDeviceInfoHandle device_info) {
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  if (impl->device_type == kMSDeviceTypeCPU || impl->device_type == kMSDeviceTypeGPU) {
    return impl->enable_fp16;
  } else {
    MS_LOG(ERROR) << "Unsupported Feature. device_type: " << impl->device_type;
    return false;
  }
}

void MSDeviceInfoSetFrequency(MSDeviceInfoHandle device_info, int frequency) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  if (impl->device_type == kMSDeviceTypeKirinNPU) {
    impl->frequency = frequency;
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
  }
}

int MSDeviceInfoGetFrequency(const MSDeviceInfoHandle device_info) {  // only for KirinNPU
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return -1;
  }
  auto impl = static_cast<mindspore::DeviceInfoC *>(device_info);
  if (impl->device_type == kMSDeviceTypeKirinNPU) {
    return impl->frequency;
  } else {
    MS_LOG(ERROR) << "Unsupported Feature.";
    return -1;
  }
}
