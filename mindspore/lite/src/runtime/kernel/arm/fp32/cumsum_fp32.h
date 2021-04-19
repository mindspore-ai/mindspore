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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUMSUM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUMSUM_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/cumsum_parameter.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class CumSumCPUKernel : public LiteKernel {
 public:
  CumSumCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<CumSumParameter *>(op_parameter_);
  }
  ~CumSumCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoCumsum(int task_id);
  int DoCumsumInt(int task_id);

 private:
  int out_dim_ = 1;
  int axis_dim_ = 1;
  int in_dim_ = 1;
  int unit_ = 1;
  CumSumParameter *param_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUMSUM_H_
