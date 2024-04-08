/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_OPS_SEQUENCE_STACK_KERNEL_H_
#define AICPU_OPS_SEQUENCE_STACK_KERNEL_H_

#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include "common/kernel_base.h"

namespace aicpu {
class SequenceStackKernel : public KernelBase {
 public:
  SequenceStackKernel() : KernelBase("SequenceStack") {}
  ~SequenceStackKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;
  template <typename T>
  uint32_t SequenceStackTask();

  aicpuops::DataType input_data_type_{aicpuops::DataType::MS_UNKNOWN};
  size_t output_data_size_{0};
  size_t dims_behind_axis_{1};
  std::vector<int64_t> tuple_shape;
  int axis_{0};
};
}  // namespace aicpu
#endif  // AICPU_OPS_SEQUENCE_STACK_KERNEL_H_
