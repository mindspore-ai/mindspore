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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FILL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FILL_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "nnacl/base/fill_base.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class FillCPUKernel : public LiteKernel {
 public:
  FillCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {}
  ~FillCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoFill(int task_id);

 private:
  int thread_sz_count_ = 0;
  int thread_sz_stride_ = 0;
  int data_size_;
  float src_data_;
  float *out_ptr_;
  int int32_src_data_;
  int *int32_out_ptr_;
  int thread_count_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FILL_H_
