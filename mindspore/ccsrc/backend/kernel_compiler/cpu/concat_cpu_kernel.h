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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CONCAT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CONCAT_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ConcatCPUKernel : public CPUKernel {
 public:
  ConcatCPUKernel() = default;
  ~ConcatCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  int axis_ = 0;
  CNodeWeakPtr node_wpt_;
};

MS_REG_CPU_KERNEL_T(
  Concat, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ConcatCPUKernel, float);
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                    ConcatCPUKernel, int8_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                    ConcatCPUKernel, int16_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    ConcatCPUKernel, int)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    ConcatCPUKernel, int64_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                    ConcatCPUKernel, uint8_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                    ConcatCPUKernel, uint16_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    ConcatCPUKernel, uint32_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                    ConcatCPUKernel, uint64_t)
MS_REG_CPU_KERNEL_T(Concat,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    ConcatCPUKernel, bool)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CONCAT_CPU_KERNEL_H_
