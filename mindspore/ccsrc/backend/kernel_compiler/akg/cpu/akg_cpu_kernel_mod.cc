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

#include "backend/kernel_compiler/akg/cpu/akg_cpu_kernel_mod.h"

#include <dlfcn.h>
#include <omp.h>
#include <thread>
#include <algorithm>
#include <memory>
#include <utility>
#include "nlohmann/json.hpp"
#include "backend/kernel_compiler/common_utils.h"
#include "common/thread_pool.h"
#include "utils/ms_utils.h"
#include "mindspore/ccsrc/debug/common.h"

namespace mindspore {
namespace kernel {
class AkgParallelLaunch {
 public:
  using AkgParallelLambda = int (*)(int task_id, int num_task, void *cdata);
  static int AkgLaunchFunc(AkgParallelLambda flambda, void *cdata, int num_task) {
    auto nthreads = omp_get_max_threads();
#pragma omp parallel num_threads(nthreads)
    { flambda(omp_get_thread_num(), nthreads, cdata); }
    return 0;
  }
};

struct AkgCallBack {
  void *parallel_launch_func;
  void *(*malloc_func)(size_t);
  void (*free_func)(void *);

  AkgCallBack() {
    parallel_launch_func = reinterpret_cast<void *>(&AkgParallelLaunch::AkgLaunchFunc);
    malloc_func = &malloc;
    free_func = &free;
  }
  ~AkgCallBack() = default;
};

CpuKernelManagerPtr CpuKernelMod::kernelmanager_ = std::make_shared<CpuKernelManager>();

CpuKernelManager::~CpuKernelManager() {
  for (auto &cpu_func_pair : cpu_func_map_) {
    if (cpu_func_pair.second.second != nullptr) {
      (void)dlclose(cpu_func_pair.second.second);
    }
  }
}

void *CpuKernelManager::SearchFunc(const std::string &kernel_name) const {
  auto iter = cpu_func_map_.find(kernel_name);
  if (iter == cpu_func_map_.end()) {
    return nullptr;
  } else {
    return iter->second.first;
  }
}

void *CpuKernelManager::SearchFuncWithSharedLock(const std::string &kernel_name) const {
  std::shared_lock lock(mutex_);
  return SearchFunc(kernel_name);
}

void *CpuKernelManager::GetFunction(const std::string &kernel_name) {
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
  KernelMeta *bin_map = KernelMeta::GetInstance();
  auto fn_so = bin_map->kernel_meta_path();
  (void)fn_so.append(fn + ".so");
  auto handle = dlopen(fn_so.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(ERROR) << "Load " << fn_so << " failed. kernel: " << kernel_name;
    return nullptr;
  }
  auto launch_func = dlsym(handle, kernel_name.c_str());
  if (launch_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol " << kernel_name << " in " << fn_so;
    return nullptr;
  }
  cpu_func_map_[kernel_name] = std::make_pair(launch_func, handle);
  return launch_func;
}

CpuKernelMod::CpuKernelMod(const KernelPackPtr &kp) {
  auto js = nlohmann::json::parse(kp->GetJson()->contents, kp->GetJson()->contents + kp->GetJson()->len);
  kernel_name_ = js["kernelName"];
  launch_func_ = kernelmanager_->GetFunction(kernel_name_);
}

bool CpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (launch_func_ == nullptr) {
    MS_LOG(ERROR) << "GetFunction failed. kernel: " << kernel_name_;
    return false;
  }
  std::vector<void *> runtimeargs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) { return output->addr; });
  static AkgCallBack akg_callback;
  (void)runtimeargs.emplace_back(reinterpret_cast<void *>(&akg_callback));
  using AkgCpuKernelFunction = void (*)(void *);
  reinterpret_cast<AkgCpuKernelFunction>(launch_func_)(reinterpret_cast<void *>(runtimeargs.data()));
  return true;
}
}  // namespace kernel
}  // namespace mindspore
