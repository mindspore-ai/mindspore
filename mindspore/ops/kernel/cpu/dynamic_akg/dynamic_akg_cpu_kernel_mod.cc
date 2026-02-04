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
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include "nlohmann/json.hpp"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/utils/thread_pool.h"
#include "utils/ms_utils.h"
#include "utils/file_utils.h"
#include "include/utils/common.h"
#include "kernel/cpu/dynamic_akg/dynamic_akg_cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kDoubleSize = 2;

class DynamicAkgParallelLaunch {
 public:
  using DynamicAkgParallelLambda = int (*)(int task_id, int num_task, void *cdata, void *extend_data);
  static int DynamicAkgLaunchFunc(DynamicAkgParallelLambda flambda, void *cdata, void *extend_data, int task_nums) {
#pragma omp parallel num_threads(task_nums)
    { flambda(omp_get_thread_num(), task_nums, cdata, extend_data); }
    return 0;
  }
};

struct DynamicAkgCallBack {
  int (*parallel_launch_func)(DynamicAkgParallelLaunch::DynamicAkgParallelLambda, void *, void *, int);

  DynamicAkgCallBack() : parallel_launch_func(&DynamicAkgParallelLaunch::DynamicAkgLaunchFunc) {}
  ~DynamicAkgCallBack() = default;
};

DynamicAkgCpuKernelManagerPtr DynamicAkgCpuKernelMod::kernel_manager_ = std::make_shared<DynamicAkgCpuKernelManager>();

DynamicAkgCpuKernelManager::~DynamicAkgCpuKernelManager() {
  for (auto &cpu_func_pair : cpu_func_map_) {
    if (cpu_func_pair.second.second != nullptr) {
      (void)dlclose(cpu_func_pair.second.second);
    }
  }
}

void DynamicAkgCpuKernelManager::GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name,
                                                          std::string *fn_so, std::string *fn_kernel) const {
  auto config_path = GetCompilerCachePath();
  auto dso_path = config_path + std::string(kAkgKernelMeta);
  (void)dso_path.append(fn + ".so");
  if (!Common::FileExists(dso_path)) {
    MS_EXCEPTION(UnknownError) << "Get Dynamic AKG kernel failed, kernel path is[" << dso_path << "].";
  }
  *fn_so = dso_path;
  *fn_kernel = fn;
}

void *DynamicAkgCpuKernelManager::GetFunction(const std::string &kernel_name) {
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
    MS_LOG(ERROR) << "Invalid file path " << fn_so << ". kernel: " << kernel_name;
    return nullptr;
  }
  auto handle = dlopen(realfile.value().c_str(), RTLD_LAZY | RTLD_LOCAL);
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

DynamicAkgCpuKernelMod::DynamicAkgCpuKernelMod(const std::string &kernel_name) {
  kernel_name_ = kernel_name;
  launch_func_ = kernel_manager_->GetFunction(kernel_name_);
}

DynamicAkgCpuKernelMod::DynamicAkgCpuKernelMod(const KernelPackPtr &kernel_pack) {
  if (kernel_pack != nullptr) {
    auto js = kernel_pack->GetJson();
    if (js != nullptr) {
      auto parsed_js = nlohmann::json::parse(js->contents, js->contents + js->len);
      kernel_name_ = parsed_js["kernelName"];
      launch_func_ = kernel_manager_->GetFunction(kernel_name_);
    }
  }
}

int DynamicAkgCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "Start resize for DynamicAkgCpuKernelMod.";
  int ret = KernelMod::Resize(inputs, outputs);

  ndims_.clear();
  shape_list_.clear();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
  }
  MS_LOG(DEBUG) << "Done resize for DynamicAkgCpuKernelMod.";
  return ret;
}

bool DynamicAkgCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                    const std::vector<KernelTensor *> &outputs, void *) {
  if (launch_func_ == nullptr) {
    MS_LOG(ERROR) << "GetFunction failed. kernel: " << kernel_name_;
    return false;
  }

  std::vector<void *> runtimeargs;
  runtimeargs.reserve(inputs.size() + outputs.size());
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](KernelTensor *input) { return input->device_ptr(); });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](KernelTensor *output) { return output->device_ptr(); });

  if (is_dynamic_) {
    MS_LOG(INFO) << "The kernel mod deals with dynamic shape inputs.";

    auto max_length_iter = std::max_element(
      shape_list_.begin(), shape_list_.end(),
      [](const std::vector<int64_t> &a, const std::vector<int64_t> &b) { return a.size() < b.size(); });
    size_t max_length = max_length_iter->size();
    std::vector<std::vector<int64_t>> arg_size_vec;
    arg_size_vec.reserve(ndims_.size());
    for (size_t i = 0; i < ndims_.size(); i++) {
      std::vector<int64_t> arg_size;
      arg_size.push_back(0);
      (void)arg_size.insert(arg_size.end(), shape_list_[i].begin(), shape_list_[i].end());
      std::vector<int64_t> strides_(ndims_[i], 1);
      for (int j = SizeToInt(ndims_[i]) - 2; j >= 0; j--) {
        strides_[j] = strides_[j + 1] * shape_list_[i][j + 1];
      }
      (void)arg_size.insert(arg_size.end(), strides_.begin(), strides_.end());
      (void)arg_size.insert(arg_size.end(), kDoubleSize * (max_length - shape_list_[i].size()), 0);
      arg_size_vec.push_back(arg_size);
    }

    std::vector<void *> arg_size_list;
    (void)std::transform(std::begin(arg_size_vec), std::end(arg_size_vec), std::back_inserter(arg_size_list),
                         [](auto &v) { return reinterpret_cast<void *>(&v[0]); });

    using DynamicAkgCpuKernelFunction = void (*)(void *, void *);
    reinterpret_cast<DynamicAkgCpuKernelFunction>(launch_func_)(reinterpret_cast<void *>(runtimeargs.data()),
                                                                reinterpret_cast<void *>(arg_size_list.data()));
  } else {
    MS_LOG(INFO) << "The kernel mod deals with static shape inputs.";
    static DynamicAkgCallBack dyn_akg_callback = DynamicAkgCallBack();
    using StaticAkgCpuKernelFunction = void (*)(void *, void *, void *);
    auto task_nums = omp_get_max_threads();
    reinterpret_cast<StaticAkgCpuKernelFunction>(launch_func_)(reinterpret_cast<void *>(&dyn_akg_callback),
                                                               reinterpret_cast<void *>(&task_nums),
                                                               reinterpret_cast<void *>(runtimeargs.data()));
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
