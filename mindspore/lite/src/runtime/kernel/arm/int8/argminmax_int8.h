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
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "nnacl/int8/arg_min_max_int8.h"
#include "nnacl/common_func.h"
#include "include/errorcode.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class ArgMinMaxInt8CPUKernel : public LiteKernel {
 public:
  ArgMinMaxInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}

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
