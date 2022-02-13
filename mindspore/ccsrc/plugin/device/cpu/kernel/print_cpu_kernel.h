/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PrintCpuKernelMod : public NativeCpuKernelMod {
 public:
  PrintCpuKernelMod() = default;
  ~PrintCpuKernelMod() override = default;

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
                    PrintCpuKernelMod, bool)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, int8_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, int16_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, int)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, int64_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, uint8_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, uint16_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, uint32_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, uint64_t)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, float16)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, float)
MS_REG_CPU_KERNEL_T(Print,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
                    PrintCpuKernelMod, double)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_
