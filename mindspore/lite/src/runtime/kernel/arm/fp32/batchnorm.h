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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_BATCHNORM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_BATCHNORM_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/nnacl/fp32/batchnorm.h"

using mindspore::lite::Context;

namespace mindspore::kernel {
class BatchnormCPUKernel : public LiteKernel {
 public:
  BatchnormCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                     const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx)
    : LiteKernel(parameter, inputs, outputs), ctx_(ctx), thread_count_(ctx->thread_num_) {
    batchnorm_param_ = reinterpret_cast<BatchNormParameter *>(parameter);
  }
  ~BatchnormCPUKernel() override { delete batchnorm_param_; }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int tid);

 private:
  int thread_count_;
  int thread_unit_;
  int units_;
  int channel_;
  float *in_addr_;
  float *mean_addr_;
  float *var_addr_;
  float *out_addr_;
  const Context *ctx_;
  BatchNormParameter *batchnorm_param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_BATCHNORM_H_
