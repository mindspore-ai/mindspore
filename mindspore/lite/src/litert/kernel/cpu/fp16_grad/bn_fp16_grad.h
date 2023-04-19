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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_BN_FP16_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_BN_FP16_GRAD_H_

#include <vector>
#include "src/executor/kernel_exec.h"
#include "nnacl/fp32_grad/batch_norm_grad.h"

namespace mindspore::kernel {

class BNGradCPUKernelFp16 : public LiteKernel {
 public:
  explicit BNGradCPUKernelFp16(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~BNGradCPUKernelFp16() override {}
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int thread_num_ = 1;
  int stage_ = 0;
  size_t ws_size_ = 0;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_BN_FP16_GRAD_H_
