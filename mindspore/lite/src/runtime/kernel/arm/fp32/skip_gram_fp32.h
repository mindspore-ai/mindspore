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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SKIP_GRAM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SKIP_GRAM_H_

#include <vector>
#include "src/lite_kernel.h"
#include "mindspore/lite/nnacl/skip_gram_parameter.h"
#include "src/common/string_util.h"

namespace mindspore::kernel {

class SkipGramCPUKernel : public LiteKernel {
 public:
  explicit SkipGramCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), ctx_(ctx), thread_count_(ctx->thread_num_) {}
  ~SkipGramCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExcute(int task_id);

 protected:
  const lite::InnerContext *ctx_ = nullptr;
  int thread_count_ = 1;
  SkipGramParameter *skip_gram_parameter_ = nullptr;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SKIP_GRAM_H_
