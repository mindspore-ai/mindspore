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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_

#include <vector>
#include "src/train/loss_kernel.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/softmax_parameter.h"

namespace mindspore::kernel {

class SparseSoftmaxCrossEntropyWithLogitsCPUKernel : public LossKernel {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsCPUKernel(OpParameter *parameter,
                                                        const std::vector<lite::Tensor *> &inputs,
                                                        const std::vector<lite::Tensor *> &outputs,
                                                        const lite::InnerContext *ctx)
      : LossKernel(parameter, inputs, outputs, ctx) {
    param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(parameter);
  }
  ~SparseSoftmaxCrossEntropyWithLogitsCPUKernel() override {
    if (sm_params_ != nullptr) {
      delete sm_params_;
      sm_params_ = nullptr;
    }
  }

  int ForwardPostExecute(const int *labels, const float *losses, float *output) const;
  int GradPostExecute(const int *labels, const float *losses, float *grads) const;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int n_dim_;
  int element_size_;
  int input_shape_[DIMENSION_5D];
  SoftmaxCrossEntropyParameter *param;
  SoftmaxParameter *sm_params_ = nullptr;
  int inner_size_ = 1;
  int outter_size_ = 1;
  int stage_ = 0;
  int threads_ = 0;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H_
