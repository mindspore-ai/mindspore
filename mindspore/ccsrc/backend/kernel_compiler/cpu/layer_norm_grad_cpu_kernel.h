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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LAYER_NORM_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LAYER_NORM_GRAD_CPU_KERNEL_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class LayerNormGradCPUKernel : public CPUKernel {
 public:
  LayerNormGradCPUKernel() = default;
  ~LayerNormGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  void CheckParam(const CNodePtr &kernel_node);
  TypeId dtype_{kTypeUnknown};
  float eps_{1e-12};
  size_t block_num_{1};
  size_t block_size_{1};
  size_t param_num_{1};
  size_t param_size_{1};
};

MS_REG_CPU_KERNEL(LayerNormGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  LayerNormGradCPUKernel);

MS_REG_CPU_KERNEL(LayerNormGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  LayerNormGradCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LAYER_NORM_GRAD_CPU_KERNEL_H_
