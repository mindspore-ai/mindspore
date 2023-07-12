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
#include "kernel/framework_utils.h"
#include "include/common/thread_pool.h"
#include "utils/ms_utils.h"
#include "utils/file_utils.h"
#include "mindspore/ccsrc/include/common/debug/common.h"
#include "plugin/device/cpu/kernel/dynamic_akg/dynamic_akg_cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
class DynamicAkgParallelLaunch {
 public:
  using DynamicAkgParallelLambda = int (*)(int task_id, int num_task, void *cdata);
  static int DynamicAkgLaunchFunc(DynamicAkgParallelLambda flambda, void *cdata, int) {
    auto nthreads = omp_get_max_threads();
#pragma omp parallel num_threads(nthreads)
    { flambda(omp_get_thread_num(), nthreads, cdata); }
    return 0;
  }
};

struct DynamicAkgCallBack {
  int (*parallel_launch_func)(DynamicAkgParallelLaunch::DynamicAkgParallelLambda, void *, int);
  void *(*malloc_func)(size_t);
  void (*free_func)(void *);
  void *extend_data = nullptr;

  DynamicAkgCallBack()
      : parallel_launch_func(&DynamicAkgParallelLaunch::DynamicAkgLaunchFunc), malloc_func(&malloc), free_func(&free) {}
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
  (void)dso_path.append(fn + "_dyn.so");
  if (!Common::FileExists(dso_path)) {
    MS_EXCEPTION(UnknownError) << "Get Dynamic AKG kernel failed, kernel path is[" << dso_path << "].";
  }
  *fn_so = dso_path;
  *fn_kernel = fn;
}

DynamicAkgCpuKernelMod::DynamicAkgCpuKernelMod(const std::string &kernel_name) {
  kernel_name_ = kernel_name;
  launch_func_ = kernel_manager_->GetFunction(kernel_name_);
}

bool DynamicAkgCpuKernelMod::Init(const BaseOperatorPtr & /* base_operator */,
                                  const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_LOG(INFO) << "input is dynamic or not: " << is_dynamic_;
  return true;
}

int DynamicAkgCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  MS_LOG(DEBUG) << "Start resize for DynamicAkgCpuKernelMod.";
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);

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

bool DynamicAkgCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
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

  if (is_dynamic_) {
    MS_LOG(INFO) << "The kernel mod deals with dynamic shape inputs.";
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
    using StaticAkgCpuKernelFunction = void (*)(void *);
    reinterpret_cast<StaticAkgCpuKernelFunction>(launch_func_)(reinterpret_cast<void *>(runtimeargs.data()));
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
