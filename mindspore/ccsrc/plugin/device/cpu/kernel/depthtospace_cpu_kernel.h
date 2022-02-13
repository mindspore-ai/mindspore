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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DEPTHTOSPACE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DEPTHTOSPACE_CPU_KERNEL_H_

#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class DepthToSpaceCpuKernelMod : public NativeCpuKernelMod {
 public:
  DepthToSpaceCpuKernelMod() = default;
  ~DepthToSpaceCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  size_t block_size_{0};
};

MS_REG_CPU_KERNEL_T(
  DepthToSpace, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  DepthToSpaceCpuKernelMod, float);

MS_REG_CPU_KERNEL_T(
  DepthToSpace, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  DepthToSpaceCpuKernelMod, float16);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                    DepthToSpaceCpuKernelMod, int8_t);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                    DepthToSpaceCpuKernelMod, int16_t);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    DepthToSpaceCpuKernelMod, int);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    DepthToSpaceCpuKernelMod, int64_t);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                    DepthToSpaceCpuKernelMod, uint8_t);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                    DepthToSpaceCpuKernelMod, uint16_t);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    DepthToSpaceCpuKernelMod, uint32_t);

MS_REG_CPU_KERNEL_T(DepthToSpace,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                    DepthToSpaceCpuKernelMod, uint64_t);

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DEPTHTOSPACE_CPU_KERNEL_H_
