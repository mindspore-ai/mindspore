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
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "plugin/factory/ms_factory.h"
#if !defined(_WIN32)
#include <dlfcn.h>
#include "extendrt/cxx_api/dlutils.h"
#endif

namespace mindspore::kernel {
AscendKernelPlugin &AscendKernelPlugin::GetInstance() {
  static AscendKernelPlugin instance;
  return instance;
}

AscendKernelPlugin::AscendKernelPlugin() : handle_(nullptr), create_kernel_map_(nullptr), is_registered_(false) {}

void AscendKernelPlugin::Register() {
#if !defined(_WIN32)
  if (is_registered_) {
    MS_LOG(INFO) << "Create kernel map has been created.";
    return;
  }
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(this), &dl_info);
  std::string cur_so_path = dl_info.dli_fname;
  auto pos = cur_so_path.find("libmindspore-lite.so");
  if (pos == std::string::npos) {
    MS_LOG(ERROR) << "Could not find libmindspore-lite so, cur so path: " << cur_so_path;
    return;
  }
  std::string parent_dir = cur_so_path.substr(0, pos);
  std::string ascend_kernel_plugin_path;
  auto ret = FindSoPath(parent_dir, "libascend_kernel_plugin.so", &ascend_kernel_plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of libascend_kernel_plugin.so failed.";
    return;
  }
  MS_LOG(INFO) << "Find ascend kernel plugin so success, path = " << ascend_kernel_plugin_path;
  void *function = nullptr;
  ret = DLSoOpen(ascend_kernel_plugin_path, "CreateCustomAscendKernel", &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << ascend_kernel_plugin_path;
    return;
  }
  auto create_kernel_func = reinterpret_cast<std::map<std::string, KernelModFunc> *(*)(void)>(function);
  if (create_kernel_func == nullptr) {
    MS_LOG(ERROR) << "Cast CreateCustomAscendKernel failed.";
    return;
  }
  create_kernel_map_ = create_kernel_func();
  if (create_kernel_map_ == nullptr) {
    MS_LOG(ERROR) << "Create custom ascend kernel failed.";
    return;
  }
  // register
  for (auto &kernel : *create_kernel_map_) {
    static KernelRegistrar<kernel::KernelMod> ascend_kernel_reg(kernel.first, kernel.second);
  }
  is_registered_ = true;
  MS_LOG(INFO) << "Register ascend kernel plugin success.";
#endif
}

void AscendKernelPlugin::DestroyAscendKernelMap() {
#if !defined(_WIN32)
  if (handle_ == nullptr) {
    MS_LOG(DEBUG) << "Handle is nullptr.";
    return;
  }
  auto destroy_map_func =
    reinterpret_cast<void (*)(std::map<std::string, KernelModFunc> *)>(dlsym(handle_, "DestroyCustomAscendKernel"));
  if (destroy_map_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol DestroyCustomAscendKernel in ['libascend_kernel_plugin.so'].";
    return;
  }
  destroy_map_func(create_kernel_map_);
  is_registered_ = false;
#endif
}

AscendKernelPlugin::~AscendKernelPlugin() {
#if !defined(_WIN32)
  MS_LOG(DEBUG) << "~AscendKernelPlugin() begin.";
  DestroyAscendKernelMap();
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
  MS_LOG(DEBUG) << "~AscendKernelPlugin() end.";
#endif
}
}  // namespace mindspore::kernel
