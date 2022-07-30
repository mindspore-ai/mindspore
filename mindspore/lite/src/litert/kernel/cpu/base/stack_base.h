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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_STACK_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_STACK_BASE_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/stack_parameter.h"

using mindspore::lite::InnerContext;
namespace mindspore::kernel {
class StackBaseCPUKernel : public LiteKernel {
 public:
  StackBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    stack_param_ = reinterpret_cast<StackParameter *>(op_parameter_);
  }
  ~StackBaseCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int StackExecute(int task_id);

 protected:
  StackParameter *stack_param_ = nullptr;
  int axis_ = 0;
  size_t data_type_size_ = 0;
  size_t copy_size_ = 0;
  int outer_size_ = 1;
  void **all_inputs_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_STACK_BASE_H_
