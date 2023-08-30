/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSEREORDER_H_
#define AICPU_KERNELS_NORMALIZED_SPARSEREORDER_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/sparse_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {

class SparseReorderCpuKernel : public CpuKernel {
 public:
  ~SparseReorderCpuKernel() = default;

  /*
   * compute sparse reorder
   * @param ctx: cpu kernel context
   * @return uint32_t: 0->success other->failed
   */
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /*
   * valid sparse reorder param
   * @param st: sparse tensor
   * @param y_indices: y_indices tensor
   * @param y_values: y_values tensor
   * @return uint32_t: 0->success other->failed
   */
  template <typename ValueT>
  uint32_t EigenSparseReorder(const CpuKernelContext &ctx, SparseTensor &st, Tensor *y_indices, Tensor *y_values) {
    if (st.GetIndicesAndValues<int64_t, ValueT>(y_indices, y_values) != KERNEL_STATUS_OK) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (st.IndicesValid(ctx) != KERNEL_STATUS_OK) {
      st.CreateSparseTensor(y_indices, y_values, st.Shape(), st.Order());
      st.Reorder<int64_t, ValueT>();
    }
    return KERNEL_STATUS_OK;
  }

  /*
   * valid sparse reorder
   * @param st: sparse tensor
   * @param y_indices: y_indices tensor
   * @param y_values: y_values tensor
   * @return uint32_t: 0->success other->failed
   */
  uint32_t SparseReorder(const CpuKernelContext &ctx, SparseTensor &st, Tensor *y_indices, Tensor *y_values);

  /*
   * valid sparse reorder param
   * @param ctx: cpu kernel context
   * @return uint32_t: 0->success other->failed
   */
  uint32_t ValidParam(const CpuKernelContext &ctx);
};

}  // namespace aicpu
#endif
