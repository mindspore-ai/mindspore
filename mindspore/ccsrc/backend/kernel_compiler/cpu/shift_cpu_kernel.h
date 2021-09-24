/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SORT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SORT_CPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ShiftCpuKernel : public CPUKernel {
 public:
  ShiftCpuKernel() = default;
  ~ShiftCpuKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  // inputs
  int64_t periods_{0};
  T fill_value_{0};

  // slice info
  AxisIterator axisIterator_{};

  // set type to int64_t to avoid computation signed and unsigned values
  int64_t copy_src_begin_{0};
  int64_t copy_dst_begin_{0};
  int64_t copy_size_{0};

  int64_t fill_begin_{0};
  int64_t fill_size_{0};
};

MS_REG_CPU_KERNEL_T(
  Shift, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  ShiftCpuKernel, bool)

MS_REG_CPU_KERNEL_T(
  Shift,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ShiftCpuKernel, float)

MS_REG_CPU_KERNEL_T(
  Shift,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ShiftCpuKernel, double)

MS_REG_CPU_KERNEL_T(
  Shift, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ShiftCpuKernel, int32_t)

MS_REG_CPU_KERNEL_T(
  Shift, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ShiftCpuKernel, int64_t)

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SORT_CPU_KERNEL_H_
