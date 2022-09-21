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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_FILL_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_FILL_FP16_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/fp16/fill_fp16.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class FillFp16CPUKernel : public LiteKernel {
 public:
  FillFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {}
  ~FillFp16CPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoFill(int task_id);

 private:
  int thread_sz_count_ = 0;
  int thread_sz_stride_ = 0;
  int data_size_ = 0;
  float16_t *fp16_out_ptr_ = nullptr;
  float16_t fp16_src_data_ = 0.0f;
  int thread_count_ = 1;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_FILL_FP16_H_
