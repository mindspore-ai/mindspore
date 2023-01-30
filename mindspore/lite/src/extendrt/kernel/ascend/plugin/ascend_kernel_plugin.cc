/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/plugin/ascend_kernel_plugin.h"
#include <map>
#include <utility>
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "plugin/factory/ms_factory.h"
#if !defined(_WIN32)
#include <dlfcn.h>
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::kernel {
std::mutex AscendKernelPlugin::mutex_;

AscendKernelPlugin::AscendKernelPlugin() = default;

AscendKernelPlugin::~AscendKernelPlugin() {
  MS_LOG(DEBUG) << "~AscendKernelPlugin() begin.";
  Unregister();
  MS_LOG(DEBUG) << "~AscendKernelPlugin() end.";
}

Status AscendKernelPlugin::TryRegister() {
  std::lock_guard<std::mutex> lock(mutex_);
  static AscendKernelPlugin instance;
  return instance.TryRegisterInner();
}

bool AscendKernelPlugin::Register() {
  std::lock_guard<std::mutex> lock(mutex_);
  static AscendKernelPlugin instance;
  auto status = instance.TryRegisterInner();
  if (status.IsError()) {
    MS_LOG(ERROR) << status.ToString();
    return false;
  }
  MS_LOG(INFO) << "Register ascend kernel plugin success.";
  return true;
}

Status AscendKernelPlugin::TryRegisterInner() {
#if !defined(_WIN32)
  if (is_registered_) {
    return kSuccess;
  }
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(this), &dl_info);
  std::string cur_so_path = dl_info.dli_fname;
  auto pos = cur_so_path.find("libmindspore-lite.so");
  if (pos == std::string::npos) {
    MS_LOG(DEBUG) << "Could not find libmindspore-lite so, cur so path: " << cur_so_path;
    auto c_lite_pos = cur_so_path.find("_c_lite");
    if (c_lite_pos == std::string::npos) {
      return {kLiteError, "Could not find _c_lite so, cur so path: " + cur_so_path};
    }
    pos = c_lite_pos;
  }
  std::string parent_dir = cur_so_path.substr(0, pos);
  std::string ascend_kernel_plugin_path;
  auto ret = FindSoPath(parent_dir, "libascend_kernel_plugin.so", &ascend_kernel_plugin_path);
  if (ret != kSuccess) {
    return {kLiteError, "Get real path of libascend_kernel_plugin.so failed."};
  }
  MS_LOG(INFO) << "Find ascend kernel plugin so success, path = " << ascend_kernel_plugin_path;
  void *function = nullptr;
  ret = DLSoOpen(ascend_kernel_plugin_path, "CreateCustomAscendKernel", &handle_, &function);
  if (ret != kSuccess) {
    return {kLiteError, "DLSoOpen failed, so path: " + ascend_kernel_plugin_path};
  }
  auto create_kernel_func = reinterpret_cast<std::map<std::string, KernelModFunc> *(*)(void)>(function);
  if (create_kernel_func == nullptr) {
    return {kLiteError, "Cast CreateCustomAscendKernel failed."};
  }
  create_kernel_map_ = create_kernel_func();
  if (create_kernel_map_ == nullptr) {
    return {kLiteError, "Create custom ascend kernel failed."};
  }
  // register
  for (auto &kernel : *create_kernel_map_) {
    if (!kernel::Factory<kernel::KernelMod>::Instance().IsRegistered(kernel.first)) {
      kernel::Factory<kernel::KernelMod>::Instance().Register(kernel.first, std::move(kernel.second));
      register_kernels_.push_back(kernel.first);
    }
  }
  is_registered_ = true;
  return kSuccess;
#endif
}

void AscendKernelPlugin::Unregister() {
#if !defined(_WIN32)
  if (handle_ == nullptr) {
    MS_LOG(INFO) << "Handle is nullptr.";
    return;
  }
  for (auto &kernel : register_kernels_) {
    kernel::Factory<kernel::KernelMod>::Instance().UnRegister(kernel);
  }
  auto destroy_map_func =
    reinterpret_cast<void (*)(std::map<std::string, KernelModFunc> *)>(dlsym(handle_, "DestroyCustomAscendKernel"));
  if (destroy_map_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol DestroyCustomAscendKernel in ['libascend_kernel_plugin.so'].";
    return;
  }
  destroy_map_func(create_kernel_map_);
  (void)dlclose(handle_);
  handle_ = nullptr;
  is_registered_ = false;
#endif
}
}  // namespace mindspore::kernel
