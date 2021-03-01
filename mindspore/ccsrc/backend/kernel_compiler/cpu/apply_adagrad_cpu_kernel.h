/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_ADAGRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_ADAGRAD_CPU_KERNEL_H_

#include <thread>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ApplyAdagradCPUKernel : public CPUKernel {
 public:
  ApplyAdagradCPUKernel() = default;
  ~ApplyAdagradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
              const std::vector<AddressPtr> &outputs) override;

 private:
  static void CheckParam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchApplyAdagrad(T const var, T const accum, const T lr, const T gradient, size_t start, size_t end);
  bool update_slots_{true};
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(ApplyAdagrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  ApplyAdagradCPUKernel);

MS_REG_CPU_KERNEL(ApplyAdagrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  ApplyAdagradCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif
