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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_ADA_FACTOR_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_ADA_FACTOR_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class FusedAdaFactorCPUKernel : public CPUKernel {
 public:
  FusedAdaFactorCPUKernel() = default;
  ~FusedAdaFactorCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  void InitInputOutputSize(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckInputAddresses(const std::vector<AddressPtr> &inputs) const;
  void CheckWorkspaceAddresses(const std::vector<AddressPtr> &workspaces) const;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                    const std::vector<AddressPtr> &outputs);

  template <typename T>
  float CalcRMS(T *input, size_t elem_num);

  template <typename T>
  void FactorUpdate(float *update, const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces);

  bool enable_scale_parameter_{false};
  bool enable_first_moment_{false};
  bool enable_weight_decay_{false};
  bool need_factor_{false};
  size_t elem_num_{0};
  size_t last_row_dim_size_{1};
  size_t last_col_dim_size_{1};
  TypeId param_dtype_{kTypeUnknown};
  float global_norm_reciprocal_{1.0f};

  enum InputEnum {
    EPSILON,
    CLIP_THRESHOLD,
    BETA1,
    BETA2T,
    WEIGHT_DECAY,
    LEARNING_RATE,
    GRAD,
    PARAM,
    EXP_AVG,
    EXP_AVG_SQ_ROW,
    EXP_AVG_SQ_COL,
    EXP_AVG_SQ,
    GLOBAL_NORM
  };

  enum WorkspaceEnum { UPDATE, R_FACTOR, C_FACTOR };
};

MS_REG_CPU_KERNEL(FusedAdaFactor,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  FusedAdaFactorCPUKernel)

MS_REG_CPU_KERNEL(FusedAdaFactor,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  FusedAdaFactorCPUKernel)

MS_REG_CPU_KERNEL(FusedAdaFactorWithGlobalNorm,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  FusedAdaFactorCPUKernel)

MS_REG_CPU_KERNEL(FusedAdaFactorWithGlobalNorm,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat16),
                  FusedAdaFactorCPUKernel)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FUSED_ADA_FACTOR_CPU_KERNEL_H_
