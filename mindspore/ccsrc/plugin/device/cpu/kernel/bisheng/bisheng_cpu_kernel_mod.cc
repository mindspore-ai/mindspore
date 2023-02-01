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

#include <dlfcn.h>
#include <omp.h>
#include <thread>
#include <algorithm>
#include <memory>
#include <utility>
#include "nlohmann/json.hpp"
#include "kernel/common_utils.h"
#include "include/common/thread_pool.h"
#include "utils/ms_utils.h"
#include "utils/file_utils.h"
#include "mindspore/ccsrc/include/common/debug/common.h"
#include "plugin/device/cpu/kernel/bisheng/bisheng_cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
class BishengParallelLaunch {
 public:
  using BishengParallelLambda = int (*)(int task_id, int num_task, void *cdata);
  static int BishengLaunchFunc(BishengParallelLambda flambda, void *cdata, int) {
    auto nthreads = omp_get_max_threads();
#pragma omp parallel num_threads(nthreads)
    { flambda(omp_get_thread_num(), nthreads, cdata); }
    return 0;
  }
};

struct BishengCallBack {
  int (*parallel_launch_func)(BishengParallelLaunch::BishengParallelLambda, void *, int);
  void *(*malloc_func)(size_t);
  void (*free_func)(void *);
  void *extend_data = nullptr;

  BishengCallBack()
      : parallel_launch_func(&BishengParallelLaunch::BishengLaunchFunc), malloc_func(&malloc), free_func(&free) {}
  ~BishengCallBack() = default;
};

BishengCpuKernelManagerPtr BishengCpuKernelMod::kernel_manager_ = std::make_shared<BishengCpuKernelManager>();

BishengCpuKernelManager::~BishengCpuKernelManager() {
  for (auto &cpu_func_pair : cpu_func_map_) {
    if (cpu_func_pair.second.second != nullptr) {
      (void)dlclose(cpu_func_pair.second.second);
    }
  }
}

void *BishengCpuKernelManager::SearchFunc(const std::string &kernel_name) const {
  auto iter = cpu_func_map_.find(kernel_name);
  if (iter == cpu_func_map_.end()) {
    return nullptr;
  } else {
    return iter->second.first;
  }
}

void *BishengCpuKernelManager::SearchFuncWithSharedLock(const std::string &kernel_name) const {
  std::shared_lock lock(mutex_);
  return SearchFunc(kernel_name);
}

void *BishengCpuKernelManager::GetFunction(const std::string &kernel_name) {
  if (auto func = SearchFuncWithSharedLock(kernel_name); func != nullptr) {
    return func;
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
  auto config_path = GetCompilerCachePath();
  auto fn_so = config_path + std::string(kBishengKernelMeta);
  (void)fn_so.append(fn + "_bisheng.so");

  if (!Common::FileExists(fn_so)) {
    MS_EXCEPTION(UnknownError) << "Get Bisheng kernel failed, kernel path is[" << fn_so << "].";
  }

  auto realfile = FileUtils::GetRealPath(fn_so.c_str());
  if (!realfile.has_value()) {
    MS_LOG(ERROR) << "Invalid file path " << fn_so << ". kernel: " << kernel_name;
    return nullptr;
  }
  auto handle = dlopen(realfile.value().c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(ERROR) << "Load " << fn_so << " failed. kernel: " << kernel_name;
    return nullptr;
  }
  auto kernel_name_wo_suffix = fn;
  auto launch_func = dlsym(handle, kernel_name_wo_suffix.c_str());
  if (launch_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol " << kernel_name_wo_suffix << " in " << fn_so;
    return nullptr;
  }
  cpu_func_map_[kernel_name] = std::make_pair(launch_func, handle);
  return launch_func;
}

BishengCpuKernelMod::BishengCpuKernelMod(const std::string &kernel_name) {
  kernel_name_ = kernel_name;
  launch_func_ = kernel_manager_->GetFunction(kernel_name_);
}

bool BishengCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs, void *) {
  if (launch_func_ == nullptr) {
    MS_LOG(ERROR) << "GetFunction failed. kernel: " << kernel_name_;
    return false;
  }

  std::vector<void *> runtimeargs;
  runtimeargs.reserve(inputs.size() + outputs.size());
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) { return output->addr; });

  using BishengCpuKernelFunction = void (*)(void *);
  reinterpret_cast<BishengCpuKernelFunction>(launch_func_)(reinterpret_cast<void *>(runtimeargs.data()));

  return true;
}
}  // namespace kernel
}  // namespace mindspore
