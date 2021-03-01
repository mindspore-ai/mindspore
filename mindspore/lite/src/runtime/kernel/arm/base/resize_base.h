/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_RESIZE_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_RESIZE_BASE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/resize_parameter.h"

using mindspore::schema::PrimitiveType_Resize;
using mindspore::schema::ResizeMethod;

namespace mindspore::kernel {
class ResizeBaseCPUKernel : public LiteKernel {
 public:
  ResizeBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}

  ~ResizeBaseCPUKernel() override = default;

  int Init() override;
  int ReSize() override { return 0; };

 protected:
  int method_ = 0;
  int new_height_ = 0;
  int new_width_ = 0;
  int coordinate_transform_mode_;
  bool preserve_aspect_ratio_ = false;
  bool const_shape_ = false;

 private:
  int CheckParameters();
  int CheckInputsOuputs();
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_RESIZE_BASE_H_
