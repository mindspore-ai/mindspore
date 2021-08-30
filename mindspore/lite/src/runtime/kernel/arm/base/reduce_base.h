/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_REDUCE_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_REDUCE_BASE_H_

#include <vector>
#include "src/inner_kernel.h"

#include "nnacl/reduce_parameter.h"

namespace mindspore::kernel {
class ReduceBaseCPUKernel : public InnerKernel {
 public:
  ReduceBaseCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(param, inputs, outputs, ctx) {}
  virtual ~ReduceBaseCPUKernel() = default;

  int Init() override;
  int ReSize() override;

 protected:
  int CheckInputsOutputs();
  int CheckParameters();

  void CalculateTmpBufferSize();
  void CalculateInnerOuterSize();

  int axes_[MAX_SHAPE_SIZE] = {0};
  int num_axes_{0};
  int mode_{0};
  bool reduce_to_end_{false};

  std::vector<size_t> buffer_sizes_;
  std::vector<int> outer_sizes_;
  std::vector<int> inner_sizes_;
  std::vector<int> axis_sizes_;
  int outer_size_{0};
  int inner_size_{0};
  int axis_size_{0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_REDUCE_BASE_H_
