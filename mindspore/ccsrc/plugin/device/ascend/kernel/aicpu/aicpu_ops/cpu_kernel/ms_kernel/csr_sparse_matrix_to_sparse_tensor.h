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
#ifndef AICPU_KERNELS_NORMALIZED_CSRSPARSEMATRIXTOSPARSETENSOR_H_
#define AICPU_KERNELS_NORMALIZED_CSRSPARSEMATRIXTOSPARSETENSOR_H_

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class CSRSparseMatrixToSparseTensorCpuKernel : public CpuKernel {
 public:
  ~CSRSparseMatrixToSparseTensorCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename indicesT, typename dataT>
  uint32_t ComputeKernel(CpuKernelContext &ctx);

  template <typename indicesT>
  void SpecialCompute(int64_t batch_begin, int64_t batch_end, CpuKernelContext &ctx);

  template <typename indicesT>
  void IndicesCompute(CpuKernelContext &ctx, int64_t indices_offset, const int64_t batch_idx, const int64_t row_idx,
                      const int64_t col_idx);
};
}  // namespace aicpu
#endif
