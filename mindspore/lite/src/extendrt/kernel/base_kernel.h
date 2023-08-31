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
#include "src/infer/primitive_type.h"
#include "src/extendrt/graph_compiler/infershape_helper.h"

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
    auto base_operator = primitive_.base_operator;
    if (base_operator != nullptr) {
      type_ = PrimitiveType(primitive_.base_operator->name());
    }
  }

  BaseKernel(InferPrimitive primitive, std::vector<InferTensor *> in_tensors, std::vector<InferTensor *> out_tensors,
             const InferContext *ctx)
      : primitive_(std::move(primitive)),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        context_(ctx) {
    auto base_operator = primitive_.base_operator;
    if (base_operator != nullptr) {
      type_ = PrimitiveType(primitive_.base_operator->name());
    }
  }

  int Prepare() override { return kLiteError; }

  int InferShape() override { return lite::NodeFallBackInferShape(primitive_.cnode, NCHW); }

  int Execute() override {
    auto ret = PreProcess();
    if (lite::RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel PreProcess failed, name: " << this->name();
      return ret;
    }

    ret = Run();
    if (lite::RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << this->name();
      return ret;
    }

    ret = PostProcess();
    if (lite::RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel PostProcess failed, name: " << this->name();
      return ret;
    }
    return lite::RET_OK;
  }

  virtual int Run() { return lite::RET_OK; }

  virtual bool InferShapeDone() const {
    auto checker = context_ != nullptr ? context_->get_infer_checker() : lite::InferCheckerOutput;
    return checker != nullptr && checker(in_tensors_, out_tensors_);
  }

  virtual int PreProcess() {
    if (!InferShapeDone()) {
      auto ret = InferShape();
      if (ret != 0) {
        MS_LOG(ERROR) << "InferShape fail!";
        return ret;
      }
      ret = ReSize();
      if (ret != 0) {
        MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
        return ret;
      }
    }

    for (auto *output : this->out_tensors()) {
      MS_ASSERT(output != nullptr);
      auto ret = output->MallocData();
      if (ret != lite::RET_OK) {
        MS_LOG(ERROR) << "MallocData failed";
        return ret;
      }
      output->ResetRefCount();
    }
    return lite::RET_OK;
  }
  // called after Run
  virtual int PostProcess() {
    for (auto &in_tensor : this->in_tensors()) {
      MS_ASSERT(in_tensor != nullptr);
      in_tensor->DecRefCount();
    }
    return lite::RET_OK;
  }

  int ReSize() override { return lite::RET_ERROR; }

  const std::vector<mindspore::MSTensor> &inputs() override;

  const std::vector<mindspore::MSTensor> &outputs() override;

  virtual void set_in_tensors(const std::vector<InferTensor *> &in_tensors) { this->in_tensors_ = in_tensors; }

  virtual void set_in_tensor(InferTensor *in_tensor, size_t index);

  virtual void set_out_tensors(const std::vector<InferTensor *> &out_tensors) { this->out_tensors_ = out_tensors; }

  virtual void set_out_tensor(InferTensor *out_tensor, size_t index);

  virtual const std::vector<InferTensor *> &in_tensors() const { return in_tensors_; }

  virtual const std::vector<InferTensor *> &out_tensors() const { return out_tensors_; }

  PrimitiveType type() { return type_; }

  virtual OpParameter *op_parameter() const { return nullptr; }

 protected:
  InferPrimitive primitive_;
  PrimitiveType type_;
  std::vector<InferTensor *> in_tensors_;
  std::vector<InferTensor *> out_tensors_;
  const InferContext *context_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_EXTENDRT_KERNEL_H_
