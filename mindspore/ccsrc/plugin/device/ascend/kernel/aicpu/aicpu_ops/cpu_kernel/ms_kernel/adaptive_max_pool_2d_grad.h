/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_ADAPTIVE_MAX_POOL2D_GRAD_H
#define AICPU_KERNELS_NORMALIZED_ADAPTIVE_MAX_POOL2D_GRAD_H

#include "inc/ms_cpu_kernel.h"
#include "cpu_types.h"
namespace aicpu {
class AdaptiveMaxPool2dGrad : public CpuKernel {
 public:
  AdaptiveMaxPool2dGrad() = default;
  ~AdaptiveMaxPool2dGrad() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename SCALAR_T>
  uint32_t DoCompute(CpuKernelContext &ctx, DataType indices_type);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_ADAPTIVE_MAX_POOL2D_GRAD_H
