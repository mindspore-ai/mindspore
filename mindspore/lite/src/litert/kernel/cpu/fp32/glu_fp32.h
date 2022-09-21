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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GLU_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GLU_FP32_H_

#include <cstring>
#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/op_base.h"
#include "nnacl/split_parameter.h"
#include "nnacl/glu_parameter.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
constexpr size_t kSplitNum = 2;

class GluCPUKernel : public LiteKernel {
 public:
  GluCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
               const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    glu_param_ = reinterpret_cast<GluParameter *>(op_parameter_);
    split_ptr_.resize(kSplitNum, nullptr);
  }
  ~GluCPUKernel() override { FreeTmpBuffer(); }

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int Split(int task_id) const;
  int Sigmoid(int task_id) const;
  int Mul(int task_id) const;

 private:
  void FreeTmpBuffer();
  int MallocTmpBuffer();

 private:
  SplitParameter split_param_{};
  GluParameter *glu_param_ = nullptr;
  void *input_ptr_ = nullptr;
  int8_t *sigmoid_ptr_ = nullptr;
  std::vector<void *> split_ptr_;
  int split_sizes_[kSplitNum] = {0};
  int thread_n_stride_ = 0;
  int usable_thread_num_ = 0;
  int num_unit_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GLU_FP32_H_
