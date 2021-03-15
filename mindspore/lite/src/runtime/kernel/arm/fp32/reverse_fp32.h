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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class ReverseCPUKernel : public LiteKernel {
 public:
  ReverseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~ReverseCPUKernel() override {
    if (tmp_ != nullptr) {
      free(tmp_);
      tmp_ = nullptr;
    }
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int Stride(int index);
  int DoReverse(int task_id);

 private:
  void UpdateAxisInfo();
  int thread_sz_count_ = 0;
  int thread_sz_stride_ = 0;
  int data_size_ = 0;
  int strides_[COMM_SHAPE_SIZE] = {0};
  int inCount_[COMM_SHAPE_SIZE] = {0};
  int outCount_[COMM_SHAPE_SIZE] = {0};
  int *tmp_ = nullptr;
  float *in_ptr_ = nullptr;
  float *out_ptr_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_H_
