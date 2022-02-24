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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ONE_HOT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ONE_HOT_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class OneHotCpuKernelMod : public NativeCpuKernelMod {
 public:
  OneHotCpuKernelMod() = default;
  ~OneHotCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename ID, typename OD>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  TypeId input_dtype_{kTypeUnknown};
  TypeId output_dtype_{kTypeUnknown};
  size_t depth_{0};
  size_t stride_{0};
  size_t axis_{0};
};

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddOutputAttr(kNumberTypeUInt8),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddOutputAttr(kNumberTypeUInt16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeUInt32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeUInt64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeInt8)
                    .AddInputAttr(kNumberTypeInt8)
                    .AddOutputAttr(kNumberTypeInt8),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddOutputAttr(kNumberTypeInt16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddOutputAttr(kNumberTypeFloat64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeBool)
                    .AddInputAttr(kNumberTypeBool)
                    .AddOutputAttr(kNumberTypeBool),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeComplex64)
                    .AddInputAttr(kNumberTypeComplex64)
                    .AddOutputAttr(kNumberTypeComplex64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeComplex128)
                    .AddInputAttr(kNumberTypeComplex128)
                    .AddOutputAttr(kNumberTypeComplex128),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kObjectTypeString)
                    .AddInputAttr(kObjectTypeString)
                    .AddOutputAttr(kObjectTypeString),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddOutputAttr(kNumberTypeUInt8),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddOutputAttr(kNumberTypeUInt16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeUInt32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeUInt64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt8)
                    .AddInputAttr(kNumberTypeInt8)
                    .AddOutputAttr(kNumberTypeInt8),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddOutputAttr(kNumberTypeInt16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddOutputAttr(kNumberTypeFloat64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeBool)
                    .AddInputAttr(kNumberTypeBool)
                    .AddOutputAttr(kNumberTypeBool),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeComplex64)
                    .AddInputAttr(kNumberTypeComplex64)
                    .AddOutputAttr(kNumberTypeComplex64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeComplex128)
                    .AddInputAttr(kNumberTypeComplex128)
                    .AddOutputAttr(kNumberTypeComplex128),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kObjectTypeString)
                    .AddInputAttr(kObjectTypeString)
                    .AddOutputAttr(kObjectTypeString),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddOutputAttr(kNumberTypeUInt8),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddOutputAttr(kNumberTypeUInt16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeUInt32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeUInt64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt8)
                    .AddInputAttr(kNumberTypeInt8)
                    .AddOutputAttr(kNumberTypeInt8),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddOutputAttr(kNumberTypeInt16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddOutputAttr(kNumberTypeFloat64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeBool)
                    .AddInputAttr(kNumberTypeBool)
                    .AddOutputAttr(kNumberTypeBool),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeComplex64)
                    .AddInputAttr(kNumberTypeComplex64)
                    .AddOutputAttr(kNumberTypeComplex64),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeComplex128)
                    .AddInputAttr(kNumberTypeComplex128)
                    .AddOutputAttr(kNumberTypeComplex128),
                  OneHotCpuKernelMod);

MS_REG_CPU_KERNEL(OneHot,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kObjectTypeString)
                    .AddInputAttr(kObjectTypeString)
                    .AddOutputAttr(kObjectTypeString),
                  OneHotCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ONE_HOT_CPU_KERNEL_H_
