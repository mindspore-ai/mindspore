/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_INNER_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_INNER_KERNEL_H_

#include <utility>
#include <vector>
#include <memory>

#include "src/tensor.h"
#include "include/errorcode.h"
#include "include/api/kernel.h"
#include "src/litert/inner_context.h"
// #include "include/api/context.h"
#include "kernel/kernel.h"
#include "extendrt/mindir_loader/abstract_kernel.h"
#include "src/extendrt/utils/tensor_utils.h"

using mindspore::infer::Abstractkernel;

namespace mindspore::kernel {
class InnerKernel : public Abstractkernel {
 public:
  InnerKernel() = default;

  InnerKernel(std::shared_ptr<mindspore::kernel::KernelMod> kernel_mod,
              mindspore::kernel::BaseOperatorPtr base_operator, std::vector<lite::Tensor *> in_tensors,
              std::vector<lite::Tensor *> out_tensors, const lite::InnerContext *ctx)
      : kernel_mod_(kernel_mod),
        base_operator_(base_operator),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        ms_context_(ctx) {}

  virtual ~InnerKernel() {}

  int Prepare() override;

  int Execute() override;

  int ReSize() override;

  int Train() override { return mindspore::lite::RET_OK; }

  bool IsTrain() const override { return true; }

  int Eval() override { return mindspore::lite::RET_OK; }

  bool IsEval() const override { return true; }

  void SetTrainable(bool trainable = true) override {}

  bool IsTrainable() const override { return true; }

  void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) override { this->in_tensors_ = in_tensors; }

  void set_in_tensor(lite::Tensor *in_tensor, size_t index) override {
    if (index >= in_tensors_.size()) {
      MS_LOG(ERROR) << "index: " << index << " larger than in_tensors size: " << in_tensors_.size();
      return;
    }
    this->in_tensors_[index] = in_tensor;
  }

  void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) override { this->out_tensors_ = out_tensors; }

  void set_out_tensor(lite::Tensor *out_tensor, size_t index) override {
    if (index >= out_tensors_.size()) {
      MS_LOG(ERROR) << "index: " << index << " larger than out_tensors size: " << out_tensors_.size();
      return;
    }
    this->out_tensors_[index] = out_tensor;
  }

  const std::vector<lite::Tensor *> &in_tensors() const override { return in_tensors_; }

  const std::vector<lite::Tensor *> &out_tensors() const override { return out_tensors_; }

 private:
  std::shared_ptr<mindspore::kernel::KernelMod> kernel_mod_ = nullptr;
  BaseOperatorPtr base_operator_ = nullptr;
  std::vector<lite::Tensor *> in_tensors_;
  std::vector<lite::Tensor *> out_tensors_;
  const mindspore::lite::InnerContext *ms_context_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_INNER_KERNEL_H_
