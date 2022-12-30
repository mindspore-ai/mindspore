/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_MATRIX_EXP_H_
#define AICPU_KERNELS_NORMALIZED_MATRIX_EXP_H_

#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"
namespace aicpu {
class MatrixExpCpuKernel : public CpuKernel {
 public:
  MatrixExpCpuKernel() = default;
  ~MatrixExpCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t MatrixExpCheck(CpuKernelContext &ctx);

  template <typename Derived1, typename Derived2, typename Derived3>
  void MTaylorApproximant(const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &I, int order,
                          Eigen::MatrixBase<Derived3> &E);

  template <typename Derived1, typename Derived2>
  void MexpImpl(const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &I,
                Eigen::MatrixBase<Derived1> &mexp, CpuKernelContext &ctx);

  template <typename T>
  uint32_t MatrixExpCompute(CpuKernelContext &ctx);

  void TyepChangeForFp16(int64_t i, int64_t m, Eigen::half *input_x, Eigen::half *output_y, CpuKernelContext &ctx);

  template <typename T>
  uint32_t MatrixExpDiffTypeCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
