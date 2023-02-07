/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_SSPADDMM_H_
#define AICPU_KERNELS_NORMALIZED_SSPADDMM_H_
#include <deque>
#include <set>
#include <unordered_map>
#include <vector>

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"

namespace aicpu {

class SspaddmmCpuKernel : public CpuKernel {
 public:
  SspaddmmCpuKernel() = default;
  uint32_t ValidParam(CpuKernelContext &ctx);
  virtual ~SspaddmmCpuKernel() = default;

  uint32_t cnt_ = 0;
  const uint32_t kInputNum = 9;
  const uint32_t kOutputNum = 3;
  const int64_t kParallelDataNumSameShape_ = 14 * 1024;
  const int64_t kParallelDataNumSameShapeMid_ = 7 * 1024;
  template <typename T>
  void Clear(Tensor *tensor, CpuKernelContext &ctx);
  template <typename T>
  void ClearIndices(Tensor *tensor, CpuKernelContext &ctx);
  template <typename T1>
  uint32_t BoundaryCheck(Tensor *, Tensor *, int64_t, CpuKernelContext &);
  template <typename T>
  uint32_t SspaddmmCompute(CpuKernelContext &ctx);
  template <typename T_idx, typename T>
  uint32_t SparseAddSparse(CpuKernelContext &ctx, Tensor *input_indices_tensor, T *in_val_addr,
                           Tensor *output_indices_tensor, Tensor *output_values_tensor);
  template <typename T_idx, typename T>
  uint32_t SparseMulDense(CpuKernelContext &ctx, Tensor *mat1_indices_tensor, T *mat1_val_addr,
                          Tensor *mat2_values_tensor, Tensor *output_indices_tensor, Tensor *output_values_tensor,
                          const int64_t row, const int64_t col);
  template <typename T>
  T *ScalarSparseMul(CpuKernelContext &ctx, Tensor *vals, Tensor *scalar);
  int64_t GetIndicesNum(Tensor *tensor);

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};  // namespace CpuKernel
};  // namespace aicpu

#endif