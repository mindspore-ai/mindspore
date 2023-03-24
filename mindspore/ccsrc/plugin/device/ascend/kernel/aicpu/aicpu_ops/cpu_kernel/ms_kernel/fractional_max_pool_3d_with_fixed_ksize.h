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

#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL3D_WITH_FIXED_KSIZE_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL3D_WITH_FIXED_KSIZE_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalMaxPool3DWithFixedKsizeCpuKernel : public CpuKernel {
 public:
  FractionalMaxPool3DWithFixedKsizeCpuKernel() = default;
  ~FractionalMaxPool3DWithFixedKsizeCpuKernel() override = default;

 private:
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  uint32_t Compute(CpuKernelContext &ctx) override;
  template <typename scalar_t, typename random_sample_t, typename argmax_t>
  uint32_t FractionalMaxPool3DWithFixedKsizeOutCpuTemplate(CpuKernelContext &ctx);
  template <typename scalar_t, typename random_sample_t>
  uint32_t DoComputeWithArgmaxType(CpuKernelContext &ctx, DataType argmax_type);
  template <typename scalar_t>
  uint32_t DoComputeWithRandomSamplesType(CpuKernelContext &ctx, DataType random_samples_type);
  std::vector<int64_t> input_shape;
  std::vector<int64_t> random_samples_shape;
  std::vector<int64_t> output_size;
  std::vector<float> kernel_size;
};
}  // namespace aicpu
#endif
