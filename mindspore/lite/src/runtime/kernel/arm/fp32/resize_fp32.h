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
#include <algorithm>
#include "include/errorcode.h"
#include "nnacl/fp32/resize_fp32.h"
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/resize_base.h"

namespace mindspore::kernel {
class ResizeCPUKernel : public ResizeBaseCPUKernel {
 public:
  ResizeCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ResizeBaseCPUKernel(parameter, inputs, outputs, ctx) {}

  ~ResizeCPUKernel() { FreeTmpBuffer(); }

  int Init() override;
  int ReSize() override;
  int Run() override;
  virtual int RunImpl(int task_id);
  int MallocTmpBuffer();
  void FreeTmpBuffer();

 protected:
  int *y_tops_ = nullptr;
  int *y_bottoms_ = nullptr;
  int *x_lefts_ = nullptr;
  int *x_rights_ = nullptr;
  float *y_bottom_weights_ = nullptr;
  float *x_left_weights_ = nullptr;
  float *line_buffer_ = nullptr;
  CalculateOriginalCoordinate calculate_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RESIZE_H_
