/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include <Eigen/Dense>
#include <array>
#include <iostream>

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class DiagonalCpuKernel final : public CpuKernel {
 public:
  DiagonalCpuKernel() = default;
  ~DiagonalCpuKernel() override = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

  template <typename T>
  uint32_t DoComputeType(CpuKernelContext &ctx);

  template <typename T>
  void set_output(int64_t *ar, T *dptr, T *y_dptr);

 private:
  uint32_t ComputeWithType(CpuKernelContext &ctx);

 private:
  int64_t offset_ = 0;
  int64_t dim1_ = 0;
  int64_t dim2_ = 1;
  int64_t dsize = 0;
};
}  // namespace aicpu
