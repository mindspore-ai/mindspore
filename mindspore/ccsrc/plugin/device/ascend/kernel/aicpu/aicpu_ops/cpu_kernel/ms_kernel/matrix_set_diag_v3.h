/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#ifndef _MATRIX_SET_DIAG_V3_KERNELS_H_
#define _MATRIX_SET_DIAG_V3_KERNELS_H_

#include "cpu_kernel/inc/cpu_attr_value.h"
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "cpu_kernel/inc/cpu_tensor.h"
#include "cpu_kernel/inc/cpu_tensor_shape.h"
#include "cpu_kernel/inc/cpu_types.h"

namespace aicpu {

class MatrixSetDiagV3CpuKernel : public CpuKernel {
 public:
  MatrixSetDiagV3CpuKernel() = default;
  ~MatrixSetDiagV3CpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckParam(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
