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
  auto ret =
    DLSoPath({"libmindspore-lite.so", "_c_lite", "tools/converter/lib"}, kAscendkernelPluginSoNmae, &plugin_path_);
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
void *AscendAllocatorPlugin::Malloc(size_t size) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return nullptr;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    return nullptr;
  }
  return ascend_allocator_plugin_impl_->Malloc(size);
#endif
  return nullptr;
}

void AscendAllocatorPlugin::Free(void *device_data) {
#if !defined(_WIN32)
  if (!is_registered_) {
    MS_LOG(ERROR) << "AscendAllocatorPlugin is not registered.";
    return;
  }
  if (ascend_allocator_plugin_impl_ == nullptr) {
    return;
  }
  if (device_data == nullptr) {
    MS_LOG(INFO) << "device data is nullptr.";
    return;
  }
  ascend_allocator_plugin_impl_->Free(device_data);
#endif
  return;
}

Status AscendAllocatorPlugin::CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size) {
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
  return ascend_allocator_plugin_impl_->CopyDeviceDataToHost(device_data, host_data, data_size);
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
