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

#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_ACTIVATION_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_ACTIVATION_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/nnacl/fp32/activation.h"
#include "src/runtime/kernel/arm/nnacl/int8/relux_int8.h"

namespace mindspore::kernel {
class ReluXInt8CPUKernel : public LiteKernel {
 public:
  ReluXInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                     const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                     const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    type_ = (reinterpret_cast<ActivationParameter *>(parameter))->type_;
  }
  ~ReluXInt8CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

  ReluXQuantArg quant_arg_;

 private:
  int type_;
};

class ReluInt8CPUKernel : public ReluXInt8CPUKernel {
 public:
  ReluInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                    const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                    const mindspore::lite::PrimitiveC *primitive)
      : ReluXInt8CPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~ReluInt8CPUKernel() override = default;

  int Init() override {
    auto ret = ReluXInt8CPUKernel::Init();
    quant_arg_.quantized_output_min = quant_arg_.output_arg.zp_;
    quant_arg_.quantized_output_max = CHAR_MAX;
    return ret;
  };
};

class Relu6Int8CPUKernel : public ReluXInt8CPUKernel {
 public:
  Relu6Int8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                     const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                     const mindspore::lite::PrimitiveC *primitive)
      : ReluXInt8CPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~Relu6Int8CPUKernel() override = default;

  int Init() override {
    auto ret = ReluXInt8CPUKernel::Init();
    quant_arg_.quantized_output_min = QuantizeToInt8(0, quant_arg_.output_arg.scale_, quant_arg_.output_arg.zp_);
    quant_arg_.quantized_output_max = QuantizeToInt8(6, quant_arg_.output_arg.scale_, quant_arg_.output_arg.zp_);
    return ret;
  };
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_ACTIVATION_H_
