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
#ifndef AICPU_OPS_SLICE_GRAD_KERNEL_H_
#define AICPU_OPS_SLICE_GRAD_KERNEL_H_

#include <vector>
#include "common/kernel_base.h"

namespace aicpu {
class SliceGradKernel : public KernelBase {
 public:
  SliceGradKernel() : KernelBase("SliceGradKernel") {}
  ~SliceGradKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;
  bool CheckParams() const;
  bool CheckBeginSizeValue();

  template <typename T, typename S>
  uint32_t SliceGradTask();

  std::vector<int64_t> dy_shape_;
  std::vector<int64_t> begin_shape_;
  std::vector<int64_t> size_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> begin_value_;
  std::vector<int64_t> size_value_;
  aicpuops::DataType dy_type_{aicpuops::DataType::MS_UNKNOWN};
  aicpuops::DataType begin_type_{aicpuops::DataType::MS_UNKNOWN};
};
}  // namespace aicpu
#endif  // AICPU_OPS_SLICE_GRAD_KERNEL_H_
