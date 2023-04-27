/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_H_

#include <utility>
#include <memory>
#include <vector>
#include "include/api/kernel_api.h"
#include "ops/base_operator.h"
#include "ir/anf.h"
#include "include/api/status.h"
#include "src/infer/tensor.h"
#include "src/infer/context.h"
#include "src/extendrt/kernel/primitive_type.h"

namespace mindspore::kernel {
using BaseOperatorPtr = std::shared_ptr<ops::BaseOperator>;
using InferContext = mindspore::infer::abstract::Context;
using InferTensor = mindspore::infer::abstract::Tensor;
struct InferPrimitive {
  BaseOperatorPtr base_operator{nullptr};
  CNodePtr cnode{nullptr};
};

class BaseKernel : public IKernel<ops::BaseOperator> {
 public:
  BaseKernel(InferPrimitive primitive, const InferContext *ctx) : primitive_(std::move(primitive)), context_(ctx) {
    type_ = PrimitiveType(primitive_.cnode->type_name());
  }

  int Prepare() override { return kLiteError; }

  int Execute() override { return kLiteError; }

  int ReSize() override { return kLiteError; }

  const std::vector<mindspore::MSTensor> &inputs() override;

  const std::vector<mindspore::MSTensor> &outputs() override;

  void set_in_tensors(const std::vector<InferTensor *> &in_tensors) { this->in_tensors_ = in_tensors; }

  void set_in_tensor(InferTensor *in_tensor, size_t index);

  void set_out_tensors(const std::vector<InferTensor *> &out_tensors) { this->out_tensors_ = out_tensors; }

  void set_out_tensor(InferTensor *out_tensor, size_t index);

  const std::vector<InferTensor *> &in_tensors() const { return in_tensors_; }

  const std::vector<InferTensor *> &out_tensors() const { return out_tensors_; }

  PrimitiveType type() { return type_; }

 protected:
  InferPrimitive primitive_;
  PrimitiveType type_;
  std::vector<InferTensor *> in_tensors_;
  std::vector<InferTensor *> out_tensors_;
  const InferContext *context_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_H_
