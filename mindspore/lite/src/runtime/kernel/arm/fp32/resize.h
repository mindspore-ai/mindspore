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
#include "src/runtime/kernel/arm/opclib/resize.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::schema::PrimitiveType_Resize;
using mindspore::schema::ResizeMethod;

namespace mindspore::kernel {
class ResizeCPUKernel : public LiteKernel {
 public:
  ResizeCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                  const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx)
      : LiteKernel(parameter, inputs, outputs), context_(ctx) {}

  ~ResizeCPUKernel() {
    if (exec_input_data_ != nullptr) {
      free(exec_input_data_);
      exec_input_data_ = nullptr;
    }
  }

  int Init() override;
  int ReSize() override { return 0; };
  int Run() override;
  int RunImpl(int task_id);

 protected:
  const lite::Context *context_;

 private:
  int CheckParameters();
  int CheckInputsOuputs();

 private:
  ResizeMethod method_;
  int64_t new_height_;
  int64_t new_width_;
  bool align_corners_;
  bool preserve_aspect_ratio;
  LayoutConvertor layout_convertor_ = nullptr;
  float *exec_input_data_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_RESIZE_H_

