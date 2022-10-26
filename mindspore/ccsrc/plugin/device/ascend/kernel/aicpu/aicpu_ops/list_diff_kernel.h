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
#ifndef AICPU_OPS_AICPU_LIST_DIFF_KERNELS_H_
#define AICPU_OPS_AICPU_LIST_DIFF_KERNELS_H_

#include <vector>
#include "common/kernel_base.h"

namespace aicpu {
class ListDiffKernel : public KernelBase {
 public:
  ListDiffKernel() : KernelBase("ListDiffKernel") {}
  ~ListDiffKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;

  template <typename T, typename Tidx>
  uint32_t ListDiffTask();
  aicpuops::DataType input_type_{aicpuops::DataType::MS_UNKNOWN};
  aicpuops::DataType idx_type_{aicpuops::DataType::MS_UNKNOWN};
  size_t x_size_{0};
  size_t y_size_{0};
  int64_t out_size_{0};
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_LIST_DIFF_KERNELS_H_
