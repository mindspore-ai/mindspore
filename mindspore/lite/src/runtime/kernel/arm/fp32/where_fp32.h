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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_WHERE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_WHERE_H_

#include <vector>
#include "src/lite_kernel.h"

#include "include/context.h"
#include "mindspore/lite/nnacl/fp32/where_fp32.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class WhereCPUKernel : public LiteKernel {
 public:
  WhereCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), ctx_(ctx), thread_count_(ctx->thread_num_) {
    where_param_ = reinterpret_cast<WhereParameter *>(op_parameter_);
  }
  ~WhereCPUKernel() = default;

  int Init() override;
  int PreProcess() override;
  int ReSize() override { return 0; }
  int Run() override;
  int RunWithSingleInput();
  int RunWithTripleInputs();
  int DoExcute(int task_id);

 protected:
  const InnerContext *ctx_;
  int thread_count_;
  WhereParameter *where_param_;

 private:
  bool *condition_;
  float *x_;
  float *y_;
  float *output_data_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_WHERE_H_
