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
#ifndef AICPU_OPS_CONCAT_OFFSET_KERNEL_H_
#define AICPU_OPS_CONCAT_OFFSET_KERNEL_H_

#include <vector>
#include <random>
#include "common/kernel_base.h"

namespace aicpu {
class ConcatOffsetKernel : public KernelBase {
 public:
  ConcatOffsetKernel() : KernelBase("ConcatOffsetKernel") {}
  ~ConcatOffsetKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;
  bool CheckParams();
  uint32_t ConcatOffsetTask();

  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<int64_t> output_shape_;
  int64_t axis_{0};
};
}  // namespace aicpu
#endif  // AICPU_OPS_CONCAT_OFFSET_KERNEL_H_
