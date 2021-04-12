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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ISFINITE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ISFINITE_CPU_KERNEL_H_

#include <vector>
#include <map>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class IsFiniteCPUKernel : public CPUKernel {
 public:
  IsFiniteCPUKernel() = default;
  ~IsFiniteCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernelNode) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernelFloat(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  void LaunchKernelOther(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  void LaunchKernelFloat16(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

 private:
  std::map<TypeId, size_t> dtype_map_ = {{kNumberTypeBool, sizeof(bool)},       {kNumberTypeInt8, sizeof(int8_t)},
                                         {kNumberTypeInt16, sizeof(int16_t)},   {kNumberTypeInt32, sizeof(int32_t)},
                                         {kNumberTypeInt64, sizeof(int64_t)},   {kNumberTypeFloat16, sizeof(float16)},
                                         {kNumberTypeFloat32, sizeof(float)},   {kNumberTypeFloat64, sizeof(double)},
                                         {kNumberTypeUInt8, sizeof(uint8_t)},   {kNumberTypeUInt16, sizeof(uint16_t)},
                                         {kNumberTypeUInt32, sizeof(uint32_t)}, {kNumberTypeUInt64, sizeof(uint64_t)}};
  TypeId input_dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);

MS_REG_CPU_KERNEL(IsFinite, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
                  IsFiniteCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ISFINITE_CPU_KERNEL_H_
