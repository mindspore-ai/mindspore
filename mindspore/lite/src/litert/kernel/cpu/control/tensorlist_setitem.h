/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_CONTROL_TENSORLIST_SETITEM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_CONTROL_TENSORLIST_SETITEM_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "src/tensorlist.h"
#include "schema/model_generated.h"
#include "nnacl/tensorlist_parameter.h"

namespace mindspore::kernel {
class TensorListSetItemCPUKernel : public LiteKernel {
 public:
  TensorListSetItemCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~TensorListSetItemCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int IncrementOutputSize(int origin_size);
  bool InferShapeDone() const override { return false; }

 private:
  int CheckParam() const;
  lite::TensorList *input0_ = nullptr;
  lite::Tensor *input2_ = nullptr;
  lite::TensorList *output0_ = nullptr;
  int index_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_CONTROL_TENSORLIST_SETITEM_H_
