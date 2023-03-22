/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL3D_GRAD_WITH_FIXED_KSIZE_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL3D_GRAD_WITH_FIXED_KSIZE_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalMaxPool3DGradWithFixedKsizeCpuKernel : public CpuKernel {
 public:
  FractionalMaxPool3DGradWithFixedKsizeCpuKernel() = default;
  ~FractionalMaxPool3DGradWithFixedKsizeCpuKernel() override = default;

 protected:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  uint32_t Compute(CpuKernelContext &ctx) override;
  template <typename backprop_t, typename argmax_t>
  uint32_t FractionalMaxPool3DGradWithFixedKsizeOutCpuTemplate(CpuKernelContext &ctx);
  template <typename backprop_t>
  uint32_t DoComputeWithArgmaxType(CpuKernelContext &ctx, DataType argmax_type);
  std::vector<int64_t> input_shape;
  std::vector<int64_t> out_backprop_shape;
  std::vector<int64_t> argmax_shape;
};
}  // namespace aicpu
#endif
