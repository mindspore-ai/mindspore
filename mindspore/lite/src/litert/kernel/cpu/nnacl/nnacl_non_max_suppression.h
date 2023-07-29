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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_NON_MAX_SUPPRESSION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_NON_MAX_SUPPRESSION_H_

#include <vector>
#include "nnacl/nnacl_kernel.h"

namespace mindspore::nnacl {
class NonMaxSuppressionKernel : public NNACLKernel {
 public:
  explicit NonMaxSuppressionKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : NNACLKernel(parameter, inputs, outputs, ctx) {}
  ~NonMaxSuppressionKernel() override = default;
  int Run() override;
  int PreProcess() override;
};
}  // namespace mindspore::nnacl

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_NNACL_NON_MAX_SUPPRESSION_H_
