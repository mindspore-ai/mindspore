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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_BASE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_BASE_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/nnacl/optimized_kernel.h"

namespace mindspore::kernel {
class ConvolutionBaseFP16CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionBaseFP16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                               const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                               const lite::Primitive *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~ConvolutionBaseFP16CPUKernel() override = default;

  int Init() override { return RET_OK; }
  int ReSize() override { return RET_OK; }
  int Run() override { return RET_OK; }
  int RunImpl(int task_id) { return RET_OK; }
  virtual int GetExecuteTensor();
  virtual int GetExecuteFilter();
  virtual void IfCastOutput();

 protected:
  float16_t *fp16_input_ = nullptr;
  float16_t *fp16_weight_ = nullptr;
  float16_t *fp16_out_ = nullptr;
  float16_t *execute_input_;
  float16_t *execute_weight_;
  float16_t *execute_output_;
  TypeId out_data_type_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_BASE_FP16_H_
