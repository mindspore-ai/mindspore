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

#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_mod.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <omp.h>
#include <thread>
#include <algorithm>
#include <memory>
#include <utility>
#include "nlohmann/json.hpp"
#include "kernel/framework_utils.h"
#include "include/common/thread_pool.h"
#include "utils/ms_utils.h"
#include "utils/file_utils.h"
#include "mindspore/ccsrc/include/common/debug/common.h"

namespace mindspore {
namespace kernel {
class AkgParallelLaunch {
 public:
  using AkgParallelLambda = int (*)(int task_id, int num_task, void *cdata);
  static int AkgLaunchFunc(AkgParallelLambda flambda, void *cdata, int) {
    auto nthreads = omp_get_max_threads();
#pragma omp parallel num_threads(nthreads)
    { flambda(omp_get_thread_num(), nthreads, cdata); }
    return 0;
  }
};

struct AkgCallBack {
  int (*parallel_launch_func)(AkgParallelLaunch::AkgParallelLambda, void *, int);
  void *(*malloc_func)(size_t);
  void (*free_func)(void *);
  void *extend_data = nullptr;

  AkgCallBack() : parallel_launch_func(&AkgParallelLaunch::AkgLaunchFunc), malloc_func(&malloc), free_func(&free) {}
};

AkgCpuKernelManagerPtr AkgCpuKernelMod::kernel_manager_ = std::make_shared<AkgCpuKernelManager>();

void AkgCpuKernelManager::GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name,
                                                   std::string *fn_so, std::string *fn_kernel) const {
  KernelMeta *bin_map = KernelMeta::GetInstance();
  auto dso_path = bin_map->kernel_meta_path();
  (void)dso_path.append(fn + ".o");
  *fn_so = dso_path;
  *fn_kernel = kernel_name;
}

void *AkgCpuKernelManager::GetFunction(const std::string &kernel_name) {
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
  std::string fn_so;
  std::string fn_kernel;
  GetFunctionAndKernelName(fn, kernel_name, &fn_so, &fn_kernel);
  auto realfile = FileUtils::GetRealPath(fn_so.c_str());
  if (!realfile.has_value()) {
    MS_LOG(ERROR) << "Invalid file path " << fn_so << " kernel: " << kernel_name;
    return nullptr;
  }
  auto akg_fd = open((*realfile).c_str(), O_RDONLY);
  struct stat sb;
  if (akg_fd < 0) {
    MS_LOG(ERROR) << "open " << (*realfile) << " failed.";
    return nullptr;
  }
  if (fstat(akg_fd, &sb) == -1) {
    MS_LOG(ERROR) << "fstat " << (*realfile) << " failed.";
    return nullptr;
  }
  auto akg_mmap = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, akg_fd, 0);
  if (akg_mmap == nullptr) {
    MS_LOG(ERROR) << "mmap " << (*realfile) << " failed.";
  }
  if (!object_loader.LoadAkgLib(akg_mmap)) {
    MS_LOG(ERROR) << "parse " << (*realfile) << " failed.";
    return nullptr;
  }
  auto launch_func = object_loader.LookupFunction(kernel_name);
  if (launch_func == nullptr) {
    MS_LOG(ERROR) << "Undefined symbol " << kernel_name << " in " << (*realfile);
    return nullptr;
  }
  (void)close(akg_fd);
  (void)munmap(akg_mmap, sb.st_size);
  cpu_func_map_[kernel_name] = std::make_pair(launch_func, nullptr);
  return launch_func;
}

AkgCpuKernelMod::AkgCpuKernelMod(const KernelPackPtr &kp) {
  auto js = nlohmann::json::parse(kp->GetJson()->contents, kp->GetJson()->contents + kp->GetJson()->len);
  kernel_name_ = js["kernelName"];
  launch_func_ = kernel_manager_->GetFunction(kernel_name_);
}

bool AkgCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                             const std::vector<AddressPtr> &outputs, void *) {
  if (launch_func_ == nullptr) {
    MS_LOG(ERROR) << "GetFunction failed. kernel: " << kernel_name_;
    return false;
  }
  static AkgCallBack akg_callback = AkgCallBack();
  std::vector<void *> runtimeargs;
  runtimeargs.reserve(inputs.size() + outputs.size() + 1);
  (void)runtimeargs.emplace_back(reinterpret_cast<void *>(&akg_callback));
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) { return output->addr; });
  using AkgCpuKernelFunction = void (*)(void *);
  reinterpret_cast<AkgCpuKernelFunction>(launch_func_)(reinterpret_cast<void *>(runtimeargs.data()));
  return true;
}
}  // namespace kernel
}  // namespace mindspore
