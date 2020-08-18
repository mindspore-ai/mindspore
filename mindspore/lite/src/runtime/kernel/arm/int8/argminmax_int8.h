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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARGMINMAX_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARGMINMAX_INT8_H_

#include <vector>
#include "src/runtime/kernel/arm/base/arg_min_max_base.h"
#include "src/runtime/kernel/arm/nnacl/quantization/quantize.h"

namespace mindspore::kernel {
class ArgMinMaxInt8CPUKernel : public ArgMinMaxBaseCPUKernel {
 public:
  ArgMinMaxInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                         const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                         const mindspore::lite::PrimitiveC *primitive)
      : ArgMinMaxBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~ArgMinMaxInt8CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  QuantArg in_quant_arg_;
  QuantArg out_quant_arg_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_ARGMINMAX_INT8_H_
