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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONCAT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONCAT_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/base/concat_base.h"
#include "nnacl/concat_parameter.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/thread_pool.h"
#include "include/context.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class ConcatCPUKernel : public LiteKernel {
 public:
  ConcatCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    concat_param_ = reinterpret_cast<ConcatParameter *>(op_parameter_);
  }

  ~ConcatCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int DoConcat(int task_id);
  int Run() override;

 private:
  ConcatParameter *concat_param_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONCAT_H_
