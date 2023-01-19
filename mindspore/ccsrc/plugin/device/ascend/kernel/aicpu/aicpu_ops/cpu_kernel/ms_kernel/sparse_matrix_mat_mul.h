/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSEMATRIXSPARSEMATMUL_H_
#define AICPU_KERNELS_NORMALIZED_SPARSEMATRIXSPARSEMATMUL_H_

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "cpu_ops_kernel.h"

namespace aicpu {

class SparseMatrixMatMulCpuKernel : public CpuKernel {
 public:
  ~SparseMatrixMatMulCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ValidParam(CpuKernelContext &ctx);
  // check if the matrix can mul
  template <typename T>
  uint32_t CheckMatMul(CpuKernelContext &ctx);
  // create eigen sparsematrix with eigen::map
  template <typename indiceT, typename valueT>
  Eigen::Ref<const Eigen::SparseMatrix<valueT, Eigen::RowMajor, indiceT> > CreateEigenSparseMatrix(
    indiceT rows, indiceT cols, int64_t nnz, indiceT *row_pointers, indiceT *col_indices, valueT *values,
    bool transpose, bool adjoint);
  // do the actual complute
  template <typename indiceT, typename valueT>
  uint32_t DoCompute(CpuKernelContext &ctx);
};

}  // namespace aicpu
#endif
