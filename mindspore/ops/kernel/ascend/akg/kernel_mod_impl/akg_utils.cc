/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/akg/kernel_mod_impl/akg_utils.h"

#include <dlfcn.h>
#include <string>
#include <map>
#include <mutex>
#include <unordered_map>

#include "utils/file_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"

namespace mindspore {
namespace kernel {
namespace akg {
constexpr auto kJsonSuffix = ".json";
constexpr auto kBinFileSuffix = ".so";
constexpr auto kDoBinFileSuffix = "_do";

std::atomic<uintptr_t> KernelManager::kernel_stub_gen_ = 0;
std::unordered_map<string, KernelMetaPtr> KernelManager::info_table_ = {};
std::mutex KernelManager::info_table_mutex_;

KernelManager::~KernelManager() {
  for (auto &func_info : info_table_) {
    if (func_info.second->handle_ != nullptr) {
      (void)dlclose(func_info.second->handle_);
    }
  }
}

void KernelManager::GetFunctionAndKernelName(const std::string &bin_file_name, const std::string &kernel_name,
                                             std::string *bin_file, std::string *bin_kernel) {
  KernelMeta *bin_map = KernelMeta::GetInstance();
  auto dso_path = bin_map->kernel_meta_path();
  (void)dso_path.append(bin_file_name + kBinFileSuffix);
  *bin_file = dso_path;
  *bin_kernel = kernel_name + kDoBinFileSuffix;
}

void *KernelManager::GenFuncStub(const mindspore::kernel::KernelPack &kernel_pack, bool force_reload,
                                 uint32_t *block_dim, void **handle) {
  MS_EXCEPTION_IF_NULL(block_dim);
  auto kernel = kernel_pack.GetKernel();
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Invalid kernel pack, json or kernel is nullptr.";
    return nullptr;
  }
  auto kernel_contents = kernel->contents;
  if (kernel_contents == nullptr) {
    MS_LOG(ERROR) << "Invalid kernel context, json or kernel is nullptr.";
    return nullptr;
  }
  auto kernel_json_info = kernel_pack.kernel_json_info();

  *block_dim = kernel_json_info.block_dim;
  string kernel_name = kernel_json_info.kernel_name;
  string bin_file_name = kernel_json_info.bin_file_name;

  if (!force_reload) {
    // use the cached object.
    std::lock_guard<std::mutex> lock(info_table_mutex_);
    auto iter = info_table_.find(kernel_name);
    if (iter != info_table_.end()) {
      auto kernelmeta = iter->second;
      *block_dim = kernelmeta->block_dim_;
      if (handle != nullptr) {
        *handle = kernelmeta->handle_;
      }
      return kernelmeta->launch_func_;
    }
  }

  std::string bin_file;
  std::string bin_kernel;
  GetFunctionAndKernelName(bin_file_name, kernel_name, &bin_file, &bin_kernel);
  auto real_file = FileUtils::GetRealPath(bin_file.c_str());
  if (!real_file.has_value()) {
    MS_LOG(ERROR) << "Invalid file path " << bin_file << " kernel: " << kernel_name;
    return nullptr;
  }
  auto file_path = real_file.value();
  if (file_path.empty()) {
    MS_LOG(ERROR) << "The AKG kernel file does not exist, kernel name: " << bin_kernel;
    return nullptr;
  }
  auto file_handle = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (file_handle == nullptr) {
    MS_LOG(ERROR) << "Load " << bin_file << " failed. kernel: " << kernel_name;
    return nullptr;
  }
  auto launch_func = dlsym(file_handle, bin_kernel.c_str());
  if (launch_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol " << bin_kernel << " in " << bin_file;
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(info_table_mutex_);
  info_table_[kernel_name] = std::make_shared<KernelMetaInfo>(KernelMetaInfo{*block_dim, launch_func, file_handle});
  return launch_func;
}
}  // namespace akg
}  // namespace kernel
}  // namespace mindspore
