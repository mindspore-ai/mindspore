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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESHAPE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESHAPE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ReshapeCpuKernelMod : public NativeCpuKernelMod {
 public:
  ReshapeCpuKernelMod() = default;
  ~ReshapeCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
};

MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(
  Reshape,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  ReshapeCpuKernelMod)
MS_REG_CPU_KERNEL(
  Reshape,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  ReshapeCpuKernelMod)
MS_REG_CPU_KERNEL(
  Reshape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ReshapeCpuKernelMod);

MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                  ReshapeCpuKernelMod);

MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(FlattenGrad, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                  ReshapeCpuKernelMod);

MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                  ReshapeCpuKernelMod);

MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                  ReshapeCpuKernelMod);
MS_REG_CPU_KERNEL(Squeeze, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                  ReshapeCpuKernelMod);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESHAPE_CPU_KERNEL_H_
