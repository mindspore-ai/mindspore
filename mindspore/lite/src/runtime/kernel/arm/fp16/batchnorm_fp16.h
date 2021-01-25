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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_BATCHNORM_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_BATCHNORM_FP16_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/batchnorm_fp32.h"

namespace mindspore::kernel {
class BatchnormFp16CPUKernel : public BatchnormCPUKernel {
 public:
  BatchnormFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : BatchnormCPUKernel(parameter, inputs, outputs, ctx) {}
  virtual ~BatchnormFp16CPUKernel() {}

  int Run() override;
  int InitConstTensor() override;
  int DoExecute(int task_id) override;

 private:
  void FreeInputAndOutput();
  bool is_input_fp32_ = false;
  bool is_output_fp32_ = false;
  float16_t *input_ = nullptr;
  float16_t *output_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_BATCHNORM_FP16_H_
