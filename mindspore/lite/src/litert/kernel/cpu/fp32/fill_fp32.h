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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_FILL_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_FILL_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/base/fill_base.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class FillCPUKernel : public LiteKernel {
 public:
  FillCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~FillCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoFill(int task_id);

 private:
  int thread_sz_count_ = 0;
  int thread_sz_stride_ = 0;
  int data_size_ = 0;
  float src_data_ = 0.0f;
  float *out_ptr_ = nullptr;
  int int32_src_data_ = 0;
  int *int32_out_ptr_ = nullptr;
  bool bool_src_data_ = false;
  bool *bool_out_ptr_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_FILL_FP32_H_
