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
#ifndef AICPU_KERNELS_NORMALIZED_AFFINE_GRID_H_
#define AICPU_KERNELS_NORMALIZED_AFFINE_GRID_H_

#include "inc/ms_cpu_kernel.h"
#include <Eigen/Dense>
#include <Eigen/Eigen>

namespace aicpu {
class AffineGridCpuKernel : public CpuKernel {
 public:
  AffineGridCpuKernel() = default;
  ~AffineGridCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t AffineGridCompute(CpuKernelContext &ctx);
  template <typename T>
  void SpecialCompute(bool theta_3D, Eigen::MatrixXf &all, T *data_theta, int64_t start, int64_t end,
                      int64_t result_row, T *output);
  template <typename T>
  uint32_t AffineGridCompute4D(CpuKernelContext &ctx, std::vector<int64_t> &data_out_size, bool align_corners);
  template <typename T>
  uint32_t AffineGridCompute5D(CpuKernelContext &ctx, std::vector<int64_t> &data_out_size, bool align_corners);
  template <typename T>
  uint32_t CommonCompute(CpuKernelContext &ctx, int64_t data_num, int64_t result_row, bool theta_3D,
                         Eigen::MatrixXf &all);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_AFFINE_GRID_H_
