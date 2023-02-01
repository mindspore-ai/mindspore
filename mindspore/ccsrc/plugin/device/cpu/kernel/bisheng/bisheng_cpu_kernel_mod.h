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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_MOD_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include "kernel/kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
class BishengCpuKernelManager {
 public:
  BishengCpuKernelManager() = default;
  ~BishengCpuKernelManager();

  void *GetFunction(const std::string &kernel_name);

 private:
  void *SearchFunc(const std::string &kernel_name) const;
  void *SearchFuncWithSharedLock(const std::string &kernel_name) const;

  // cache the kernel function: kernel_name -> {kernel_func, so_handle}
  std::unordered_map<std::string, std::pair<void *, void *>> cpu_func_map_;
  mutable std::shared_mutex mutex_;
};
using BishengCpuKernelManagerPtr = std::shared_ptr<BishengCpuKernelManager>;
class BishengCpuKernelMod : public CpuKernelMod {
 public:
  explicit BishengCpuKernelMod(const std::string &kernel_name);
  ~BishengCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *) override;

  std::vector<KernelAttr> GetOpSupport() { return {}; }

  static BishengCpuKernelManagerPtr kernel_manager_;

 private:
  void *launch_func_;
};

using BishengCpuKernelModPtr = std::shared_ptr<BishengCpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_MOD_H_
