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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FULLCONNECTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FULLCONNECTION_H_

#include <vector>
#include "include/errorcode.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/nnacl/fp32/matmul.h"
#include "src/runtime/kernel/arm/base/fullconnection_base.h"

using mindspore::lite::Context;

namespace mindspore::kernel {
class FullconnectionCPUKernel : public FullconnectionBaseCPUKernel {
 public:
  FullconnectionCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                          const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                          const lite::Primitive *primitive)
      : FullconnectionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~FullconnectionCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int DoMatmul(int task_id);

 private:
  float *a_c8_ptr_;
  float *b_r8_ptr_;
  float *c_r8x8_ptr_;
  float *bias_ptr_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FULLCONNECTION_H_
