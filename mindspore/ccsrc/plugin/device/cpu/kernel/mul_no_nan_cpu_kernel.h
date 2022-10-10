/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MUL_NO_NAN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MUL_NO_NAN_CPU_KERNEL_H_

#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MulNoNanCPUKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  MulNoNanCPUKernelMod() = default;
  ~MulNoNanCPUKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex64)
        .AddInputAttr(kNumberTypeComplex64)
        .AddOutputAttr(kNumberTypeComplex64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex128)
        .AddInputAttr(kNumberTypeComplex128)
        .AddOutputAttr(kNumberTypeComplex128)};
    return support_list;
  }

 private:
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> input1_shape_;
  std::vector<int64_t> output_shape_;
  TypeId input_dtype_{kTypeUnknown};
  TypeId output_dtype_{kTypeUnknown};

  template <typename T>
  void NoBcastCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void BcastCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MUL_NO_NAN_CPU_KERNEL_H_
