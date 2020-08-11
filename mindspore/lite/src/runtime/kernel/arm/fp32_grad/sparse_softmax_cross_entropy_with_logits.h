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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_

#include <vector>
#include "src/lite_kernel.h"
#include "ir/anf.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/softmax_grad.h"
#include "src/runtime/kernel/arm/nnacl/fp32/arithmetic.h"

namespace mindspore::kernel {

class SparseSoftmaxCrossEntropyWithLogitsCPUKernel : public LiteKernel {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsCPUKernel(OpParameter *parameter,
                                                        const std::vector<lite::tensor::Tensor *> &inputs,
                                                        const std::vector<lite::tensor::Tensor *> &outputs,
                                                        const lite::Context *ctx, const lite::Primitive *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(parameter);
  }
  ~SparseSoftmaxCrossEntropyWithLogitsCPUKernel() override = default;

  void ForwardPostExecute(const int *labels, const float *losses, float *output) const;
  void GradPostExecute(const int *labels, const float *losses, float *output) const;

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  SoftmaxCrossEntropyParameter *param;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
