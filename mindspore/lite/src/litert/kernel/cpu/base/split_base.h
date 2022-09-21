/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_SPLIT_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_SPLIT_BASE_H_

#include <vector>
#include "include/errorcode.h"
#include "src/litert/lite_kernel.h"
#include "nnacl/split_parameter.h"
#include "nnacl/base/split_base.h"

namespace mindspore::kernel {
class SplitBaseCPUKernel : public LiteKernel {
 public:
  SplitBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param = reinterpret_cast<SplitParameter *>(op_parameter_);
  }
  ~SplitBaseCPUKernel() override {
    if (param != nullptr && param->split_sizes_ != nullptr) {
      free(param->split_sizes_);
      param->split_sizes_ = nullptr;
    }
  }
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int Split(int task_id);
  static int CheckAndInitSplitParam(const lite::Tensor &in_tensor, SplitParameter *param);

 protected:
  int thread_n_stride_ = 0;
  int num_unit_ = 0;
  SplitParameter *param = nullptr;
  void *input_ptr_ = nullptr;
  std::vector<void *> output_ptr_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_SPLIT_BASE_H_
