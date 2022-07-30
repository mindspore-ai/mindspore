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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_FUSED_BATCHNORM_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_FUSED_BATCHNORM_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/batchnorm_fp32.h"
#include "nnacl/fp32/scale_fp32.h"

namespace mindspore::kernel {
class FusedBatchnormCPUKernel : public BatchnormCPUKernel {
 public:
  FusedBatchnormCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : BatchnormCPUKernel(parameter, inputs, outputs, ctx) {}
  ~FusedBatchnormCPUKernel() { FreeScaleAndOffset(); }

  int Eval() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int InitConstTensor() override;
  int DoExecute(int task_id) override;
  int Batchnorm2Scale(const void *scale_data, const void *bias_data, const void *mean_data, const void *var_data,
                      float eps, int kernel_num) override;

 protected:
  void FreeScaleAndOffset();
  int InitScaleParam();
  void *scale_ = nullptr;
  void *offset_ = nullptr;
  bool trained_ = false;
  bool is_scale_ = false;
  ScaleParameter *scale_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_FUSED_BATCHNORM_FP32_H_
