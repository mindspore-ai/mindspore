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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXIMUMGRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXIMUMGRAD_CPU_KERNEL_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class MaximumGradCPUKernel : public CPUKernel {
 public:
  MaximumGradCPUKernel() = default;
  ~MaximumGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  void CheckParam(const CNodePtr &kernel_node);
  std::vector<size_t> x_shape_;
  std::vector<size_t> y_shape_;
  std::vector<size_t> dout_shape;
  std::vector<size_t> dx_shape;
  std::vector<size_t> dy_shape;
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(MaximumGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  MaximumGradCPUKernel);

MS_REG_CPU_KERNEL(MaximumGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeUInt32),
                  MaximumGradCPUKernel);

MS_REG_CPU_KERNEL(MaximumGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  MaximumGradCPUKernel);

MS_REG_CPU_KERNEL(MaximumGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  MaximumGradCPUKernel);

MS_REG_CPU_KERNEL(MaximumGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeUInt64),
                  MaximumGradCPUKernel);

MS_REG_CPU_KERNEL(MaximumGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddOutputAttr(kNumberTypeFloat64)
                    .AddOutputAttr(kNumberTypeFloat64),
                  MaximumGradCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MaximumGrad_CPU_KERNEL_H_
