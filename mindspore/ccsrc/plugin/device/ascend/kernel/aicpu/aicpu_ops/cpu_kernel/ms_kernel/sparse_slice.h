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

#include <string>
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
class SparseSliceCpuKernel : public CpuKernel {
 public:
  SparseSliceCpuKernel() = default;
  ~SparseSliceCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
  uint32_t SparseSliceParamCheck(Tensor *indices, Tensor *values, Tensor *shape, Tensor *start, Tensor *size);
};
}  // namespace aicpu
#endif
