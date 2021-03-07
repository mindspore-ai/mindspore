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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_ARGMINMAX_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_ARGMINMAX_BASE_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp32/arg_min_max_fp32.h"
#ifdef ENABLE_ARM64
#include "nnacl/fp16/arg_min_max_fp16.h"
#endif
#include "nnacl/common_func.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class ArgMinMaxCPUKernel : public LiteKernel {
 public:
  ArgMinMaxCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    arg_param_ = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  }

  ~ArgMinMaxCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  ArgMinMaxParameter *arg_param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_ARGMINMAX_BASE_H_
