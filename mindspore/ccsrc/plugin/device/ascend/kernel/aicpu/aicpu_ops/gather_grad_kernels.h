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
#ifndef AICPU_OPS_AICPU_GATHER_GRAD_KERNELS_H_
#define AICPU_OPS_AICPU_GATHER_GRAD_KERNELS_H_

#include <vector>
#include "common/kernel_base.h"

namespace aicpu {
class GatherDGradV2Kernel : public KernelBase {
 public:
  GatherDGradV2Kernel() : KernelBase("GatherDGradV2Kernel") {}
  ~GatherDGradV2Kernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;

  template <typename T, typename S>
  uint32_t GatherDGradV2Task();

  int64_t dim_{0};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> index_shape_;
  std::vector<int64_t> grad_shape_;
  std::vector<int64_t> output_shape_;

  aicpuops::DataType index_type_{aicpuops::DataType::MS_UNKNOWN};
  aicpuops::DataType grad_type_{aicpuops::DataType::MS_UNKNOWN};
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_GATHER_GRAD_KERNELS_H_
