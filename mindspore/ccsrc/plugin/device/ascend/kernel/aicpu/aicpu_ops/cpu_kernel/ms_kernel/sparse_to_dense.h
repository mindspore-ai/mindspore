/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSETODENSE_H_
#define AICPU_KERNELS_NORMALIZED_SPARSETODENSE_H_

#include "cpu_ops_kernel.h"
#include "utils/sparse_tensor.h"

namespace aicpu {

class SparseToDenseCpuKernel : public CpuKernel {
 public:
  ~SparseToDenseCpuKernel() = default;

  /*
   * compute sparse to dense
   * @param ctx: cpu kernel context
   * @return uint32_t: 0->success other->failed
   */
  uint32_t Compute(CpuKernelContext &ctx) override;

 protected:
  /*
   * valid sparse to dense param
   * @param st: sparse tensor
   * @param indices: indices tensor
   * @param output: output tensor
   * @return uint32_t: 0->success other->failed
   */
  template <typename ValueT>
  uint32_t EigenSparseToDense(const CpuKernelContext &ctx, SparseTensor &st, const Tensor *indices, Tensor *output) {
    if (indices->GetDataType() == DT_INT32) {
      return st.ToDense<int32_t, ValueT>(ctx, output);
    } else {
      return st.ToDense<int64_t, ValueT>(ctx, output);
    }
  }

  /*
   * valid sparse to dense param
   * @param st: sparse tensor
   * @param indices: indices tensor
   * @param output: output tensor
   * @return uint32_t: 0->success other->failed
   */
  uint32_t SparseToDense(const CpuKernelContext &ctx, SparseTensor &st, const Tensor *indices, Tensor *output);

  /*
   * valid sparse to dense param
   * @param ctx: cpu kernel context
   * @return uint32_t: 0->success other->failed
   */
  KernelStatus ValidParam(const CpuKernelContext &ctx);

  /*
   * parallel set default value to dense
   * @param ctx: cpu kernel context
   * @param default_value_tensor: default value of dense tensor
   * @param output_tensor: output tensor
   * @param output_size: output tensor size
   * @return uint32_t: 0->success other->failed
   */
  uint32_t ParallelSetDefaultValue(const CpuKernelContext &ctx, const Tensor *default_value_tensor,
                                   const Tensor *output_tensor, int64_t output_size);

  /*
   * set default value to dense
   * @param ctx: cpu kernel context
   * @param default_value_tensor: default value of dense tensor
   * @param output_tensor: output tensor
   * @param output_size: output tensor size
   * @return uint32_t: 0->success other->failed
   */
  uint32_t SetDefaultValue(const CpuKernelContext &ctx, const Tensor *default_value_tensor, const Tensor *output_tensor,
                           int64_t output_size);
};

}  // namespace aicpu
#endif
