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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_AVG_POOL_2D_GRAD_V1_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_AVG_POOL_2D_GRAD_V1_CPU_KERNEL_H_
#include <functional>
#include <memory>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class AdaptiveAvgPool2DGradV1CpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  AdaptiveAvgPool2DGradV1CpuKernelMod() = default;
  ~AdaptiveAvgPool2DGradV1CpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<kernel::AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::vector<int64_t> grad_output_dim_sizes;
  std::vector<int64_t> orig_input_shape_dim_sizes;
  std::vector<int64_t> grad_input_dim_sizes;
  TypeId dtype_{kTypeUnknown};

  template <typename SCALAR_T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_AVG_POOL_2D_GRAD_CPU_KERNEL_H_
