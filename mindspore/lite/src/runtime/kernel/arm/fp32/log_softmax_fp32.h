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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LOG_SOFTMAX_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LOG_SOFTMAX_H_

#include <vector>
#include "src/inner_kernel.h"
#include "src/runtime/kernel/arm/base/softmax_base.h"

namespace mindspore::kernel {
class LogSoftmaxCPUKernel : public SoftmaxBaseCPUKernel {
 public:
  LogSoftmaxCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : SoftmaxBaseCPUKernel(parameter, inputs, outputs, ctx), tmp_data_(nullptr) {}
  ~LogSoftmaxCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoLogSoftmaxLastAxis(int task_id);

 private:
  float *tmp_data_ = nullptr;
  int in_plane_size_ = 0;
  int out_plane_size_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LOG_SOFTMAX_H_
