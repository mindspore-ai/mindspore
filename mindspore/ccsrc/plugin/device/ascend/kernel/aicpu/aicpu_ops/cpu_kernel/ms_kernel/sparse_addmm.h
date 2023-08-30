/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSE_ADDMM_H_
#define AICPU_KERNELS_NORMALIZED_SPARSE_ADDMM_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
class SparseAddmmCpuKernel : public CpuKernel {
 public:
  ~SparseAddmmCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t SparseAddmmCheck(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeRowAndCol1(const CpuKernelContext &ctx,
                             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> *sparse, int64_t row_x1,
                             int64_t col_x1);

  template <typename T>
  uint32_t ComputeRowAndCol2(const CpuKernelContext &ctx,
                             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> *dense, int64_t row_x2,
                             int64_t col_x2);

  template <typename T>
  uint32_t ComputeRowAndCol3(const CpuKernelContext &ctx,
                             const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &sparse,
                             const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &dense,
                             int64_t row_x1, int64_t col_x2);

  template <typename T, typename T1>
  uint32_t SparseAddmmCompute(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
