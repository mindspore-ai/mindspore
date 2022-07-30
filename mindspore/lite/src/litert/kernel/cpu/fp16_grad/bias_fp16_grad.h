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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_BIAS_FP16_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_BIAS_FP16_GRAD_H_

#include <vector>
#include "src/litert/kernel_exec.h"
#include "nnacl/fp16/arithmetic_fp16.h"

namespace mindspore::kernel {
class BiasGradCPUKernelFp16 : public LiteKernel {
 public:
  explicit BiasGradCPUKernelFp16(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    bias_param = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~BiasGradCPUKernelFp16() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  ArithmeticParameter *bias_param;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_BIAS_FP16_GRAD_H_
