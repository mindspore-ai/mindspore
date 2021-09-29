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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BARTLETT_WINDOW_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BARTLETT_WINDOW_CPU_KERNEL_H_
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class BartlettWindowCpuKernelMod : public NativeCpuKernelMod {
 public:
  BartlettWindowCpuKernelMod() = default;
  ~BartlettWindowCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  // template <typename T, typename S>
  // void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  bool periodic_{true};
  TypeId output_dtype{kNumberTypeFloat32};
  TypeId input_dtype{kTypeUnknown};
  std::vector<size_t> input_shape;
};

MS_REG_CPU_KERNEL_T_S(BartlettWindow, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                      BartlettWindowCpuKernelMod, int32_t, float);
MS_REG_CPU_KERNEL_T_S(BartlettWindow, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
                      BartlettWindowCpuKernelMod, int32_t, float16);
MS_REG_CPU_KERNEL_T_S(BartlettWindow, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
                      BartlettWindowCpuKernelMod, int32_t, double);
MS_REG_CPU_KERNEL_T_S(BartlettWindow, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
                      BartlettWindowCpuKernelMod, int64_t, float);
MS_REG_CPU_KERNEL_T_S(BartlettWindow, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
                      BartlettWindowCpuKernelMod, int64_t, float16);
MS_REG_CPU_KERNEL_T_S(BartlettWindow, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
                      BartlettWindowCpuKernelMod, int64_t, double);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BARTLETT_WINDOW_CPU_KERNEL_H_
