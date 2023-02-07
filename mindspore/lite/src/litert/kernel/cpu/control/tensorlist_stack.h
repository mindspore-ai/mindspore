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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_CONTROL_TENSORLIST_STACK_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_CONTROL_TENSORLIST_STACK_H_

#include <vector>

#include "src/litert/lite_kernel.h"
#include "src/tensorlist.h"
#include "schema/model_generated.h"
#include "nnacl/tensorlist_parameter.h"

namespace mindspore::kernel {
class TensorListStackCPUKernel : public LiteKernel {
 public:
  TensorListStackCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx),
        num_element_(reinterpret_cast<TensorListParameter *>(parameter)->num_element_),
        dtype_(static_cast<TypeId>(reinterpret_cast<TensorListParameter *>(parameter)->element_dtype_)) {}
  ~TensorListStackCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int CheckParam();
  int MergeElementShape();
  int MergeSubShape(const std::vector<int> &shape);
  bool IsFullyDefined(const std::vector<int> &shape) const;
  bool InferShapeDone() const override { return false; }

 private:
  size_t TypeUnknownSize = 0;
  int num_element_ = -1;
  TypeId dtype_ = kTypeUnknown;
  lite::TensorList *input0_ = nullptr;
  lite::Tensor *output0_ = nullptr;
  std::vector<int> output_shape_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_CONTROL_TENSORLIST_STACK_H_
