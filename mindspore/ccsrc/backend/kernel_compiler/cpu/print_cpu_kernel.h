/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PrintCPUKernel : public CPUKernel {
 public:
  PrintCPUKernel() = default;
  ~PrintCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void LaunchKernel(const std::vector<AddressPtr> &inputs);

  TypeId CheckType();

 private:
  std::vector<std::vector<size_t>> input_shapes_;
  std::vector<size_t> input_sizes_;
};

MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, bool)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, int8_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, int16_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, int)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, int64_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, uint8_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, uint16_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, uint32_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, uint64_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, float16)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                    PrintCPUKernel, float)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_
