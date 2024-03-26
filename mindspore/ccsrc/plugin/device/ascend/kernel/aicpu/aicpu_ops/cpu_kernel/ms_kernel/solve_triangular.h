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
#ifndef AICPU_KERNELS_NORMALIZED_SOLVE_TRIANGULAR_H_
#define AICPU_KERNELS_NORMALIZED_SOLVE_TRIANGULAR_H_

#include "inc/ms_cpu_kernel.h"
#include "Eigen/Dense"

namespace aicpu {
class SolveTriangularCpuKernel : public CpuKernel {
 public:
  SolveTriangularCpuKernel() = default;
  ~SolveTriangularCpuKernel() override = default;
  template <typename Derived_a, typename Derived_b, typename T>
  static inline void solve(const Eigen::MatrixBase<Derived_a> &a, const Eigen::MatrixBase<Derived_b> &b, T *output_addr,
                           int m, int n, bool lower, bool unit_diagonal) {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output(output_addr, m, n);
    if (unit_diagonal) {
      if (lower) {
        output.noalias() = a.template triangularView<Eigen::UnitLower>().solve(b);
      } else {
        output.noalias() = a.template triangularView<Eigen::UnitUpper>().solve(b);
      }
    } else {
      if (lower) {
        output.noalias() = a.template triangularView<Eigen::Lower>().solve(b);
      } else {
        output.noalias() = a.template triangularView<Eigen::Upper>().solve(b);
      }
    }
  }

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t SolveTriangularCheck(CpuKernelContext &ctx);
  template <typename T_in, typename T_out>
  uint32_t SolveTriangularCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif