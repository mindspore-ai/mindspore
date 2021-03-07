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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_PRELU_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_PRELU_INT8_H_

#include <vector>
#include <limits>
#include "include/errorcode.h"
#include "include/context.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/int8/leaky_relu_int8.h"
#include "src/lite_kernel.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
class LeakyReluInt8CPUKernel : public LiteKernel {
 public:
  LeakyReluInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~LeakyReluInt8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  LeakyReluQuantArg quant_prelu_parm_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_PRELU_INT8_H_
