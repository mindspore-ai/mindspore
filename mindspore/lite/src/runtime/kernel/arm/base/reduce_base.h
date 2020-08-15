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
#include "src/lite_kernel.h"
#include "ir/anf.h"
#include "nnacl/reduce_parameter.h"

namespace mindspore::kernel {
class ReduceBaseCPUKernel : public LiteKernel {
 public:
  ReduceBaseCPUKernel(OpParameter *param, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                      const lite::Primitive *primitive)
      : LiteKernel(param, inputs, outputs, ctx, primitive) {}
  virtual ~ReduceBaseCPUKernel() = default;

  int Init() override;
  int ReSize() override { return 0; };

 private:
  int CheckInputsOutputs();
  int CheckParameters();

 protected:
  int axes_[REDUCE_MAX_AXES_NUM];
  int num_axes_;
  int mode_;

 protected:
  int outer_size_;
  int inner_size_;
  int axis_size_;
  std::vector<int> tmp_shape_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_REDUCE_BASE_H_
