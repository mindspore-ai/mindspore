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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_ADAM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_ADAM_H_

#include <vector>
#include "src/train/optimizer_kernel.h"
#include "nnacl/fp32_grad/optimizer.h"

namespace mindspore::kernel {
constexpr int kAdamLrIndex = 5;
constexpr int kAdamGradIndex = 9;

class AdamCPUKernel : public OptimizerKernel {
 public:
  explicit AdamCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OptimizerKernel(parameter, inputs, outputs, ctx, kAdamLrIndex, kAdamGradIndex),
        thread_count_(ctx->thread_num_) {
    adam_param_ = reinterpret_cast<AdamParameter *>(parameter);
  }
  ~AdamCPUKernel() override {
    if (grad_sum_ != nullptr) {
      ms_context_->allocator->Free(grad_sum_);
      grad_sum_ = nullptr;
    }
  }
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);
  int OptimizerStep() override;
  std::vector<int> GetOptimizerParamsIdxs() const override;
  std::vector<int> GetTrainableParamsIdxs() const override;

 private:
  int thread_count_;
  AdamParameter *adam_param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_ADAM_H_
