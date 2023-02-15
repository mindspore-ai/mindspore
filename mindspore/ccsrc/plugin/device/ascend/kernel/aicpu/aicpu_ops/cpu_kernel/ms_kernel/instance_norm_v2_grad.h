/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_INSTANCE_NORM_V2_GRAD_H_
#define AICPU_KERNELS_NORMALIZED_INSTANCE_NORM_V2_GRAD_H_

#include "cpu_ops_kernel.h"
#include <vector>

namespace aicpu {
class InstanceNormV2GradCpuKernel : public CpuKernel {
 public:
  InstanceNormV2GradCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t InstanceNormV2GradParamCheck(const CpuKernelContext &ctx);
  uint32_t InstanceNormV2GradShapeCheck(const CpuKernelContext &ctx);
  uint32_t InstanceNormV2GradTypeCheck(const CpuKernelContext &ctx);
  uint32_t InstanceNormV2GradAttrCheck(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  bool is_training_ = true;
  float epsilon_ = 0.00001;
  std::vector<int64_t> dy_shape_4d_;
  std::vector<int64_t> batch_channels_2d_;
  int64_t instance_num = 0;
};
}  // namespace aicpu
#endif
