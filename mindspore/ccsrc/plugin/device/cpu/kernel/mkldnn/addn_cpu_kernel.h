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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADDN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADDN_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class AddNCpuKernelMod : public MKLCpuKernelMod {
 public:
  AddNCpuKernelMod() = default;
  ~AddNCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  size_t input_num_{0};
  std::vector<size_t> output_shape_;
  TypeId dtype_{kNumberTypeFloat32};
};

MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                    AddNCpuKernelMod, int8_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                    AddNCpuKernelMod, int16_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    AddNCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    AddNCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                    AddNCpuKernelMod, uint8_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                    AddNCpuKernelMod, uint16_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    AddNCpuKernelMod, uint32_t);
MS_REG_CPU_KERNEL_T(AddN,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                    AddNCpuKernelMod, uint64_t);
MS_REG_CPU_KERNEL_T(
  AddN, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  AddNCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(
  AddN, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  AddNCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  AddN, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  AddNCpuKernelMod, double);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADDN_CPU_KERNEL_H_
