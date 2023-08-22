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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_NNACL_NNACL_BASE_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_NNACL_NNACL_BASE_KERNEL_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "src/extendrt/kernel/base_kernel.h"
#include "src/litert/lite_kernel.h"
#include "ops/base_operator.h"

namespace mindspore::kernel {
class NNACLBaseKernel : public BaseKernel {
 public:
  explicit NNACLBaseKernel(std::shared_ptr<LiteKernel> lite_kernel)
      : BaseKernel({}, nullptr), lite_kernel_(std::move(lite_kernel)) {
    this->type_ = schema::EnumNamePrimitiveType(lite_kernel_->type());
  }
  ~NNACLBaseKernel() override = default;

  int Prepare() override { return lite_kernel_->Prepare(); }
  int InferShape() override { return lite_kernel_->InferShape(); }
  int ReSize() override { return lite_kernel_->ReSize(); }
  int Run() override { return lite::RET_ERROR; }
  int Execute() override { return lite_kernel_->Execute(); }
  const std::vector<mindspore::MSTensor> &inputs() override { return lite_kernel_->inputs(); }
  const std::vector<mindspore::MSTensor> &outputs() override { return lite_kernel_->outputs(); }
  void set_in_tensors(const std::vector<InferTensor *> &in_tensors) override {
    lite_kernel_->set_in_tensors(in_tensors);
  }
  void set_in_tensor(InferTensor *in_tensor, size_t index) override { lite_kernel_->set_in_tensor(in_tensor, index); }
  void set_out_tensors(const std::vector<InferTensor *> &out_tensors) override {
    lite_kernel_->set_out_tensors(out_tensors);
  }
  void set_out_tensor(InferTensor *out_tensor, size_t index) override {
    lite_kernel_->set_out_tensor(out_tensor, index);
  }
  const std::vector<InferTensor *> &in_tensors() const override { return lite_kernel_->in_tensors(); }
  const std::vector<InferTensor *> &out_tensors() const override { return lite_kernel_->out_tensors(); }
  OpParameter *op_parameter() const override { return lite_kernel_->op_parameter(); }

 private:
  std::shared_ptr<LiteKernel> lite_kernel_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_NNACL_NNACL_BASE_KERNEL_H_
