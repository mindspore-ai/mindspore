/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/kernel/ascend/plugin/ascend_allocator_plugin.h"
#include <memory>
#if !defined(_WIN32)
#include "src/extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::kernel {
namespace {
constexpr auto kAscendkernelPluginSoNmae = "libascend_kernel_plugin.so";
constexpr auto kFunCreateAscendAllocatorPluginImpl = "CreateAclAllocator";
#if !defined(_WIN32)
std::mutex mutex_;
#endif
}  // namespace

AscendAllocatorPlugin::AscendAllocatorPlugin() = default;

AscendAllocatorPlugin::~AscendAllocatorPlugin() {
#if !defined(_WIN32)
  std::lock_guard<std::mutex> l(mutex_);
  MS_LOG(INFO) << "AscendAllocatorPlugin::~AscendAllocatorPlugin() begin.";
  ascend_allocator_plugin_impl_ = nullptr;
  DLSoClose(handle_);
  handle_ = nullptr;
  MS_LOG(INFO) << "AscendAllocatorPlugin::~AscendAllocatorPlugin() end.";
#endif
}

AscendAllocatorPlugin &AscendAllocatorPlugin::GetInstance() {
#if !defined(_WIN32)
  std::lock_guard<std::mutex> l(mutex_);
#endif
  static AscendAllocatorPlugin instance;
  return instance;
}

bool AscendAllocatorPlugin::Register() {
#if !defined(_WIN32)
  std::lock_guard<std::mutex> l(mutex_);
  if (is_registered_) {
    return true;
  }
  MS_LOG(INFO) << "AscendAllocatorPlugin Register.";
  auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite", "tools/converter/lib", "libmindspore_converter.so"},
                      kAscendkernelPluginSoNmae, &plugin_path_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "get real path of " << kAscendkernelPluginSoNmae << " failed.";
    return false;
  }
  MS_LOG(INFO) << "find ascend allocator plugin so path: " << plugin_path_;
  void *func = nullptr;
  ret = DLSoOpen(plugin_path_, kFunCreateAscendAllocatorPluginImpl, &handle_, &func);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path_
                  << " , func name: " << kFunCreateAscendAllocatorPluginImpl << ", err: " << ret.ToString();
    return false;
  }
  auto create_plugin_impl_func = reinterpret_cast<AscendAllocatorPluginImpl *(*)(void)>(func);
  if (create_plugin_impl_func == nullptr) {
    MS_LOG(ERROR) << "cast " << kFunCreateAscendAllocatorPluginImpl << " failed.";
    return false;
  }
  ascend_allocator_plugin_impl_ = std::shared_ptr<AscendAllocatorPluginImpl>(create_plugin_impl_func());
  if (ascend_allocator_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "create ascend allocator plugin impl failed.";
    return false;
  }
  is_registered_ = true;
  MS_LOG(INFO) << "register ascend allocator success.";
#endif
  return true;
}

int AscendAllocatorPlugin::GetCurrentDeviceId() {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return -1;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    return -1;
  }
  auto device_data = ascend_allocator_plugin_impl_->GetCurrentDeviceId();
  return device_data;
#endif
  return -1;
}

void *AscendAllocatorPlugin::Malloc(size_t size, int device_id) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return nullptr;
  }
  if (device_id < -1) {
    MS_LOG(ERROR) << "device id must more than 0";
    return nullptr;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "ascend_allocator_plugin_impl_ is nullptr.";
    return nullptr;
  }
  auto device_data = ascend_allocator_plugin_impl_->Malloc(size, device_id);
  return device_data;
#endif
  return nullptr;
}

void AscendAllocatorPlugin::Free(void *device_data, int device_id) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "ascend_allocator_plugin_impl_ is nullptr.";
    return;
  }
  if (device_data == nullptr) {
    MS_LOG(ERROR) << "device data is nullptr.";
    return;
  }
  ascend_allocator_plugin_impl_->Free(device_data, device_id);
#endif
  return;
}

void *AscendAllocatorPlugin::MallocHost(size_t size) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return nullptr;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "ascend_allocator_plugin_impl_ is nullptr.";
    return nullptr;
  }
  auto device_data = ascend_allocator_plugin_impl_->MallocHost(size);
  return device_data;
#endif
  return nullptr;
}

void AscendAllocatorPlugin::FreeHost(void *host_data) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    MS_LOG(ERROR) << "ascend_allocator_plugin_impl_ is nullptr.";
    return;
  }
  if (host_data == nullptr) {
    MS_LOG(ERROR) << "host data is nullptr.";
    return;
  }
  ascend_allocator_plugin_impl_->FreeHost(host_data);
#endif
  return;
}

Status AscendAllocatorPlugin::CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size,
                                                   int device_id) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return kLiteMemoryFailed;
  }
  if (device_data == nullptr) {
    MS_LOG(INFO) << "device data is nullptr.";
    return kLiteMemoryFailed;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    return kLiteMemoryFailed;
  }
  return ascend_allocator_plugin_impl_->CopyDeviceDataToHost(device_data, host_data, data_size, device_id);
#endif
  return kSuccess;
}

Status AscendAllocatorPlugin::CopyDeviceDataToDevice(void *src_device, void *dst_device, size_t src_data_size,
                                                     size_t dst_data_size, int src_device_id, int dst_device_id) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return kLiteMemoryFailed;
  }
  if (src_device_id < -1 || dst_device_id < -1) {
    MS_LOG(ERROR) << "device id is wrong, src device id: " << src_device_id << ", dst device id: " << dst_device_id;
    return kLiteError;
  }
  if (dst_device == nullptr || src_device == nullptr) {
    MS_LOG(INFO) << "device data is nullptr.";
    return kLiteMemoryFailed;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    return kLiteMemoryFailed;
  }
  return ascend_allocator_plugin_impl_->CopyDeviceDataToDevice(src_device, dst_device, src_data_size, dst_data_size,
                                                               src_device_id, dst_device_id);
#endif
  return kSuccess;
}

Status AscendAllocatorPlugin::CopyHostDataToDevice(void *host_data, void *device_data, size_t data_size) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return kLiteMemoryFailed;
  }
  if (device_data == nullptr) {
    MS_LOG(INFO) << "device data is nullptr.";
    return kLiteMemoryFailed;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    return kLiteMemoryFailed;
  }
  return ascend_allocator_plugin_impl_->CopyHostDataToDevice(host_data, device_data, data_size);
#endif
  return kSuccess;
}
}  // namespace mindspore::kernel
