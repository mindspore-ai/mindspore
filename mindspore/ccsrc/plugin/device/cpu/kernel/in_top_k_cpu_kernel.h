/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_TOP_K_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_TOP_K_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class InTopKCpuKernelMod : public NativeCpuKernelMod {
 public:
  InTopKCpuKernelMod() = default;
  ~InTopKCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  using InTopKFunc = std::function<bool(InTopKCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, InTopKFunc>> func_list_;
  InTopKFunc kernel_func_;
  size_t outer_size_{1};
  size_t inner_size_{1};
  int64_t k_{1};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_TOP_K_CPU_KERNEL_H_
