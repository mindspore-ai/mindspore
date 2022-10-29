/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ONLINE_FUSION_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ONLINE_FUSION_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/split_parameter.h"

namespace mindspore::kernel {
class SplitReduceConcatFusionCPUKernel : public LiteKernel {
 public:
  SplitReduceConcatFusionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<SplitParameter *>(op_parameter_);
  }
  ~SplitReduceConcatFusionCPUKernel() override {
    if (param_ != nullptr && param_->split_sizes_ != nullptr) {
      free(param_->split_sizes_);
      param_->split_sizes_ = nullptr;
    }
  }

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoSplitReduceConcatFusion(int task_id);

  int *split_slices_ = nullptr;
  SplitParameter *param_ = nullptr;
  int64_t inner_size_ = 0;
  int64_t outer_size_ = 0;
  int64_t mid_size_ = 0;
  int64_t mid_len_ = 0;
  size_t axis_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_ONLINE_FUSION_FP32_H_
