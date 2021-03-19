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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SOFTMAX_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SOFTMAX_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/softmax_base.h"

namespace mindspore::kernel {
class SoftmaxFp16CPUKernel : public SoftmaxBaseCPUKernel {
 public:
  SoftmaxFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : SoftmaxBaseCPUKernel(parameter, inputs, outputs, ctx), sum_data_(nullptr) {}
  ~SoftmaxFp16CPUKernel() override {
    if (sum_data_ != nullptr) {
      free(sum_data_);
    }
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int MallocTmpBuffer();
  void FreeTmpBuffer();
  int DoSoftmaxLastAxis(int task_id);

 private:
  float16_t *sum_data_ = nullptr;
  float16_t *input_fp16_ = nullptr;
  float16_t *output_fp16_ = nullptr;
  int in_plane_size_ = 0;
  int out_plane_size_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SOFTMAX_FP16_H_
