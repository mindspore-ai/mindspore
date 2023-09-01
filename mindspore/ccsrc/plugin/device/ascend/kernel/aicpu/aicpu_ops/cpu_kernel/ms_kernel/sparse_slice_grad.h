/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
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

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
class SparseSliceGradCpuKernel : public CpuKernel {
 public:
  SparseSliceGradCpuKernel() = default;
  ~SparseSliceGradCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
  uint32_t SparseSliceGradParamCheck(Tensor *backprop_val_grad, Tensor *indices, Tensor *start, Tensor *new_indices);

  template <typename T>
  uint32_t GradCompute(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
