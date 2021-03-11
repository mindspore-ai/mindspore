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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ArithmeticSelfCPUKernel : public CPUKernel {
 public:
  ArithmeticSelfCPUKernel() = default;
  ~ArithmeticSelfCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchKernelLogic(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  OperateType operate_type_{SQUARE};
  TypeId dtype_{kTypeUnknown};
  TypeId target_dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(ZerosLike, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(OnesLike, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(OnesLike, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Sign, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Sign, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Floor, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Reciprocal, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(GeLU, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(LogicalNot, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Asin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(ACos, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Atan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Sin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Cos, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Tan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Sinh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Cosh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Asinh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Acosh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
MS_REG_CPU_KERNEL(Atanh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
