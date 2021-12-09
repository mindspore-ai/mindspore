/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADJUST_CONTRASTV2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADJUST_CONTRASTV2_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t MIN_DIM = 3;

class AdjustContrastv2CpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  AdjustContrastv2CpuKernelMod() = default;
  ~AdjustContrastv2CpuKernelMod() override = default;
  void InitKernel(const CNodePtr &Kernel_node);
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  std::uint32_t LaunchAdjustContrastv2Kernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs);
  std::vector<size_t> images_shape;
  TypeId input_type_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADJUST_CONTRASTV2_CPU_KERNEL_H_
