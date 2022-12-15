/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_TRIU_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_TRIU_FP32_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp32/crop_fp32.h"
#include "nnacl/clip_parameter.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class TriuCPUKernel : public LiteKernel {
 public:
  TriuCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~TriuCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 private:
  int64_t k_ = 0;

  int GetKValue();
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_TRIU_FP32_H_
