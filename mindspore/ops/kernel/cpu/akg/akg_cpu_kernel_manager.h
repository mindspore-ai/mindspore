/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_CPU_AKG_CPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_CPU_AKG_CPU_KERNEL_MOD_H_
#include <string>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <shared_mutex>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/cpu/cpu_kernel_mod.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/kash/kernel_pack.h"

namespace mindspore {
namespace kernel {
class AkgCpuKernelManagerAbs {
 public:
  AkgCpuKernelManagerAbs() = default;
  virtual ~AkgCpuKernelManagerAbs() = default;
  virtual void *GetFunction(const std::string &kernel_name) = 0;
  virtual void GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name, std::string *fn_so,
                                        std::string *fn_kernel) const = 0;

 protected:
  void *SearchFunc(const std::string &kernel_name) const;
  void *SearchFuncWithSharedLock(const std::string &kernel_name) const;

  // cache the kernel function: kernel_name -> {kernel_func, so_handle}
  std::unordered_map<std::string, std::pair<void *, void *>> cpu_func_map_;
  mutable std::shared_mutex mutex_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_CPU_AKG_CPU_KERNEL_MOD_H_
