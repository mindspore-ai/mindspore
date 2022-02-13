/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPPER_BOUND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPPER_BOUND_CPU_KERNEL_H_

#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"
namespace mindspore {
namespace kernel {
template <typename I, typename O>
class UpperBoundCpuKernelMod : public NativeCpuKernelMod {
 public:
  UpperBoundCpuKernelMod() = default;
  ~UpperBoundCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  std::vector<size_t> sorted_x_shape_;
  std::vector<size_t> values_shape_;
  std::vector<size_t> output_shape_;
  size_t sorted_x_num_;
  size_t values_num_;
  size_t output_num_;
};

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, float16, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, float, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, double, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, int8_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, int16_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, int32_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, int64_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, uint8_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
  UpperBoundCpuKernelMod, uint16_t, int32_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, float16, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, float, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, double, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, int8_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, int16_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, int32_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, int64_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, uint8_t, int64_t);

MS_REG_CPU_KERNEL_T_S(
  UpperBound,
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
  UpperBoundCpuKernelMod, uint16_t, int64_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPPER_BOUND_CPU_KERNEL_H_
