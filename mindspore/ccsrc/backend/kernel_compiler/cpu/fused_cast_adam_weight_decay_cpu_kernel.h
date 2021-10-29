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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_CAST_ADAM_WEIGHT_DECAY_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_CAST_ADAM_WEIGHT_DECAY_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kSizeFloat16 = sizeof(float16);
constexpr size_t kScalarIndex = 0;
constexpr size_t kFusedCastAdamWeightDecayInputNum = 9;
constexpr size_t kFusedCastAdamWeightDecayOutputNum = 3;

class FusedCastAdamWeightDecayCPUKernel : public CPUKernel {
 public:
  FusedCastAdamWeightDecayCPUKernel() = default;
  ~FusedCastAdamWeightDecayCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;
  void LaunchFusedCastAdamFp32(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  void LaunchFusedCastAdamFp16(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  size_t elem_num_{0};
  TypeId var_dtype_{kTypeUnknown};
  TypeId gradient_dtype_{kTypeUnknown};
  enum input_list_ { VAR, M, V, LR, BETA1, BETA2, EPSILON, DECAY, GRAD };
};

MS_REG_CPU_KERNEL(FusedCastAdamWeightDecay,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  FusedCastAdamWeightDecayCPUKernel)

MS_REG_CPU_KERNEL(FusedCastAdamWeightDecay,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  FusedCastAdamWeightDecayCPUKernel)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_CAST_ADAM_WEIGHT_DECAY_CPU_KERNEL_H_
