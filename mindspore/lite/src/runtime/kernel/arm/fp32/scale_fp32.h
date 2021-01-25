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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SCALE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SCALE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp32/scale_fp32.h"

namespace mindspore::kernel {

class ScaleCPUKernel : public LiteKernel {
 public:
  ScaleCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    scale_param_ = reinterpret_cast<ScaleParameter *>(op_parameter_);
  }
  ~ScaleCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int CalculateParameter();
  virtual int InitScaleOffset();
  int Scale(int task_id);

 protected:
  ScaleParameter *scale_param_;

 private:
  float *input_ptr_ = nullptr;
  float *scale_ = nullptr;
  float *offset_ = nullptr;
  float *output_ptr_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SCALE_H_
