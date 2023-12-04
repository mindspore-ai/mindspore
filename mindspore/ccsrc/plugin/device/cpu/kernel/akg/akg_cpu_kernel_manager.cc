/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_manager.h"
#include <dlfcn.h>
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
void *AkgCpuKernelManagerAbs::SearchFunc(const std::string &kernel_name) const {
  auto iter = cpu_func_map_.find(kernel_name);
  if (iter == cpu_func_map_.end()) {
    return nullptr;
  } else {
    return iter->second.first;
  }
}

void *AkgCpuKernelManagerAbs::SearchFuncWithSharedLock(const std::string &kernel_name) const {
  std::shared_lock lock(mutex_);
  return SearchFunc(kernel_name);
}

void *AkgCpuKernelManagerAbs::GetFunction(const std::string &kernel_name) {
  if (auto kernel_func = SearchFuncWithSharedLock(kernel_name); kernel_func != nullptr) {
    return kernel_func;
  }
  std::unique_lock lock(mutex_);
  // Search cache again between setting unique lock and calling "dlopen", to make sure that
  // only one thread can call "dlopen" and insert handle to the cache for a new kernel_name.
  // To avoid that several nodes (with the same kernel_name) open the same "so" by dlopen,
  // but only cache it once, then the "dlclose" will be called only once, causing resource leak.
  if (auto func = SearchFunc(kernel_name); func != nullptr) {
    return func;
  }
  std::string fn;
  auto it = kernel_name.rfind("_kernel");
  if (it < kernel_name.size()) {
    fn = kernel_name.substr(0, it);
  } else {
    fn = kernel_name;
  }
  std::string fn_so;
  std::string fn_kernel;
  GetFunctionAndKernelName(fn, kernel_name, &fn_so, &fn_kernel);
  auto realfile = FileUtils::GetRealPath(fn_so.c_str());
  if (!realfile.has_value()) {
    MS_LOG(ERROR) << "Invalid file path " << fn_so << " kernel: " << kernel_name;
    return nullptr;
  }
  auto file_path = realfile.value();
  if (file_path.empty()) {
    MS_LOG(ERROR) << "The AKG kernel file does not exist, kernel name: " << fn_kernel;
    return nullptr;
  }
  auto handle = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(ERROR) << "Load " << fn_so << " failed. kernel: " << kernel_name;
    return nullptr;
  }
  auto launch_func = dlsym(handle, fn_kernel.c_str());
  if (launch_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol " << fn_kernel << " in " << fn_so;
    return nullptr;
  }
  cpu_func_map_[kernel_name] = std::make_pair(launch_func, handle);
  return launch_func;
}

}  // namespace kernel
}  // namespace mindspore
