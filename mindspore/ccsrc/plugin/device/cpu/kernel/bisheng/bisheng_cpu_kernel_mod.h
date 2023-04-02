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
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
class BishengCpuKernelManager : public AkgCpuKernelManager {
 public:
  BishengCpuKernelManager() = default;
  ~BishengCpuKernelManager();
  void GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name, std::string *fn_so,
                                std::string *fn_kernel) const;

 private:
  // cache the kernel function: kernel_name -> {kernel_func, so_handle}
  std::unordered_map<std::string, std::pair<void *, void *>> cpu_func_map_;
};
using BishengCpuKernelManagerPtr = std::shared_ptr<BishengCpuKernelManager>;
class BishengCpuKernelMod : public CpuKernelMod {
 public:
  explicit BishengCpuKernelMod(const std::string &kernel_name);
  ~BishengCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *) override;

  std::vector<KernelAttr> GetOpSupport() override { return {}; }

  static BishengCpuKernelManagerPtr kernel_manager_;

 private:
  void *launch_func_;
};

using BishengCpuKernelModPtr = std::shared_ptr<BishengCpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_MOD_H_
