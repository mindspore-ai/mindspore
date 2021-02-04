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
#ifndef MINDSPORE_PACK_CPU_KERNEL_H
#define MINDSPORE_PACK_CPU_KERNEL_H

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PackCpuFwdKernel : public CPUKernel {
 public:
  PackCpuFwdKernel();
  ~PackCpuFwdKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  bool CheckParam(const std::vector<AddressPtr> &outputs);
  void PackTensor(T *output, size_t start, size_t end);

  int axis_;
  size_t input_num_;
  size_t output_size_;
  size_t dims_behind_axis_;
  std::unique_ptr<T *[]> inputs_host_;
};

MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                    PackCpuFwdKernel, int8_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                    PackCpuFwdKernel, int16_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    PackCpuFwdKernel, int32_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    PackCpuFwdKernel, int64_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                    PackCpuFwdKernel, uint8_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    PackCpuFwdKernel, bool)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                    PackCpuFwdKernel, uint16_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    PackCpuFwdKernel, uint32_t)
MS_REG_CPU_KERNEL_T(Stack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                    PackCpuFwdKernel, uint64_t)
MS_REG_CPU_KERNEL_T(
  Stack, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  PackCpuFwdKernel, float16)
MS_REG_CPU_KERNEL_T(
  Stack, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  PackCpuFwdKernel, float)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_PACK_CPU_KERNEL_H
