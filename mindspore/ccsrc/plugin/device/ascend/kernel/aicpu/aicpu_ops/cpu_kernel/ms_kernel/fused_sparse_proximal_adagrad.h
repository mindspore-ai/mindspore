/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_FUSED_SPARSE_PROXIMAL_ADAGRAD_H_
#define AICPU_KERNELS_NORMALIZED_FUSED_SPARSE_PROXIMAL_ADAGRAD_H_
#include "cpu_kernel_utils.h"

namespace aicpu {
class FusedSparseProximalAdagradCpuKernel : public CpuKernel {
 public:
  FusedSparseProximalAdagradCpuKernel() = default;
  ~FusedSparseProximalAdagradCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t DoCompute(CpuKernelContext &ctx);
  size_t indices_size_{0};
  size_t var_first_dim_size_{0};
  size_t var_outer_dim_size_{1};
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_FUSED_SPARSE_PROXIMAL_ADAGRAD_H_
