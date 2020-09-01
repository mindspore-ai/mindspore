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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RESIZE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RESIZE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/resize_base.h"

using mindspore::schema::PrimitiveType_Resize;
using mindspore::schema::ResizeMethod;

namespace mindspore::kernel {
class ResizeCPUKernel : public ResizeBaseCPUKernel {
 public:
  ResizeCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                  const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                  const mindspore::lite::PrimitiveC *primitive)
      : ResizeBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~ResizeCPUKernel() { FreeTmpBuffer(); }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int MallocTmpBuffer();
  void FreeTmpBuffer();

 private:
  int *y_tops_ = nullptr;
  int *y_bottoms_ = nullptr;
  int *x_lefts_ = nullptr;
  int *x_rights_ = nullptr;
  float *y_bottom_weights_ = nullptr;
  float *x_left_weights_ = nullptr;
  float *line_buffer_ = nullptr;
  float *line0_ = nullptr;
  float *line1_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RESIZE_H_
