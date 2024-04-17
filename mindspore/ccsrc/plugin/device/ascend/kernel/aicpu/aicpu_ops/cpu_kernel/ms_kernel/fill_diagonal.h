/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_FILL_DIAGONAL_H_
#define AICPU_KERNELS_NORMALIZED_FILL_DIAGONAL_H_

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class FillDiagonalCpuKernel : public CpuKernel {
 public:
  FillDiagonalCpuKernel() = default;
  ~FillDiagonalCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t FillDiag(int64_t input_dims, int64_t stride, int64_t height, int64_t width, CpuKernelContext &ctx);
  float fill_value_;
};
}  // namespace aicpu
#endif