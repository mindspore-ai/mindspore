/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_BIAS_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_BIAS_FP32_H_
#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/fp32/arithmetic_fp32.h"

namespace mindspore::kernel {
class BiasCPUKernel : public LiteKernel {
 public:
  BiasCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~BiasCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int ChooseThreadCuttingStrategy();
  bool batch_priority_{false};
  int64_t inner_num_{0};
  int64_t outer_num_{0};
  int64_t total_num_{0};
  std::vector<int64_t> split_points_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_BIAS_FP32_H_
