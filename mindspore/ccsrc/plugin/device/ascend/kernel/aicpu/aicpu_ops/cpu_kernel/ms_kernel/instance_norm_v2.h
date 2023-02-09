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

#ifndef AICPU_KERNELS_NORMALIZED_INSTANCE_NORM_V2_H_
#define AICPU_KERNELS_NORMALIZED_INSTANCE_NORM_V2_H_

#include "cpu_ops_kernel.h"
#include <vector>
#include "utils/eigen_tensor.h"

namespace aicpu {
class InstanceNormV2CpuKernel : public CpuKernel {
 public:
  InstanceNormV2CpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t InstanceNormV2ParamCheck(const CpuKernelContext &ctx);
  uint32_t InstanceNormV2ShapeCheck(const CpuKernelContext &ctx);
  uint32_t InstanceNormV2TypeCheck(const CpuKernelContext &ctx);
  uint32_t InstanceNormV2AttrCheck(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t CollectStatsKernel(const CpuKernelContext &ctx, float *_mean_, float *_var_sum);

  uint32_t CollectLinearAndConstant(const CpuKernelContext &ctx, const typename TTypes<float>::Vec &gamma,
                                    const typename TTypes<float>::Vec &beta,
                                    const typename TTypes<float>::Vec &running_mean,
                                    const typename TTypes<float>::Vec &running_var,
                                    const typename TTypes<float>::Vec &save_mean,
                                    const typename TTypes<float>::Vec &save_invstd, float *_alpha_, float *_beta_);

  template <typename T>
  uint32_t TransformInput(const CpuKernelContext &ctx);

  template <typename T, template <typename S> class VarTransform>
  uint32_t UpdateStatsTemplate(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  bool is_training_ = true;
  float momentum_ = 0.1;
  float epsilon_ = 0.00001;
  std::vector<int64_t> x_shape_4d_;
  std::vector<int64_t> batch_channels_2d_;
  int64_t instance_num = 0;
};
}  // namespace aicpu
#endif
