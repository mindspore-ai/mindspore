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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_RESIZE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_RESIZE_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/resize_base.h"
#include "nnacl/quantization/quantize.h"

using mindspore::schema::PrimitiveType_Resize;
using mindspore::schema::ResizeMethod;

namespace mindspore::kernel {
class ResizeInt8CPUKernel : public ResizeBaseCPUKernel {
 public:
  ResizeInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                      const lite::Primitive *primitive)
      : ResizeBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~ResizeInt8CPUKernel() {
    delete quant_out_;
    quant_out_ = nullptr;
    delete quant_in_;
    quant_in_ = nullptr;
    delete multiplier_;
    multiplier_ = nullptr;
  }

  int Init() override;
  int ReSize() override { return 0; };
  int Run() override;
  int RunImpl(int task_id);

 private:
  QuantArg *quant_in_;
  QuantArg *quant_out_;
  QuantMulArg *multiplier_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_RESIZE_INT8_H_
