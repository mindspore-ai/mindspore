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
#include <string>
#include <vector>
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

void BishengCpuKernelManager::GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name,
                                                       std::string *fn_so, std::string *fn_kernel) const {
  auto config_path = GetCompilerCachePath();
  auto dso_path = config_path + std::string(kBishengKernelMeta);
  (void)dso_path.append(fn + "_bisheng.so");
  if (!Common::FileExists(dso_path)) {
    MS_EXCEPTION(UnknownError) << "Get Bisheng kernel failed, kernel path is[" << dso_path << "].";
  }
  *fn_so = dso_path;
  *fn_kernel = fn;
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
