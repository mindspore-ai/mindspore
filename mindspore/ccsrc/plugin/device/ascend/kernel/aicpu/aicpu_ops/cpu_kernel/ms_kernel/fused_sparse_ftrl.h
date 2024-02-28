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

#ifndef MS_KERNEL_FUSED_SPARSE_FTRL_KERNELS_H_
#define MS_KERNEL_FUSED_SPARSE_FTRL_KERNELS_H_

#include "inc/ms_cpu_kernel.h"
#include <algorithm>
#include <cmath>

namespace aicpu {
class FusedSparseFtrlKernel : public CpuKernel {
 public:
  explicit FusedSparseFtrlKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  size_t indices_size_{0};
  size_t var_first_dim_size_{0};
  size_t var_outer_dim_size_{1};
  float lr_{0};
  float l1_{0};
  float l2_{0};
  float lr_power_{0};

  uint32_t ParseKernelParam(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
