/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CROP_AND_RESIZE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CROP_AND_RESIZE_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/resize_parameter.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class CropAndResizeCPUKernel : public LiteKernel {
 public:
  CropAndResizeCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<CropAndResizeParameter *>(op_parameter_);
  }

  ~CropAndResizeCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 protected:
  int MallocTmpBuffer();
  void FreeTmpBuffer();

  CropAndResizeParameter *param_;
  int batch_;
  int new_height_;
  int new_width_;
  int *y_tops_ = nullptr;
  int *y_bottoms_ = nullptr;
  int *x_lefts_ = nullptr;
  int *x_rights_ = nullptr;
  float *y_bottom_weights_ = nullptr;
  float *x_left_weights_ = nullptr;
  float *line_buffer_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CROP_AND_RESIZE_H_
