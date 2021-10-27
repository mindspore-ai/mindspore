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

#ifndef MINDSPORE_LITE_SRC_INNER_KERNEL_H_
#define MINDSPORE_LITE_SRC_INNER_KERNEL_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "src/inner_context.h"
#include "src/tensor.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/cxx_api/tensor/tensor_impl.h"
#include "include/api/context.h"
#include "include/api/kernel.h"

namespace mindspore::kernel {
class InnerKernel : public Kernel {
 public:
  InnerKernel() = default;

  InnerKernel(OpParameter *parameter, std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
              const lite::Context *ctx)
      : op_parameter_(parameter),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        ms_context_(ctx) {}

  virtual ~InnerKernel() {
    if (op_parameter_ != nullptr) {
      free(op_parameter_);
      op_parameter_ = nullptr;
      FreeWorkspace();
    }
  }

  int Execute() override;

  virtual int Run() { return mindspore::lite::RET_ERROR; }
  int ReSize() override { return mindspore::lite::RET_ERROR; }

  // called before Run
  virtual int PreProcess();
  // called after Run
  virtual int PostProcess() { return FreeInWorkTensor(); }

  virtual bool CheckInputsValid() const { return true; }

  virtual int FreeInWorkTensor() const {
    for (auto &in_tensor : this->in_tensors()) {
      MS_ASSERT(in_tensor != nullptr);
      in_tensor->DecRefCount();
    }
    return lite::RET_OK;
  }

  int Prepare() override { return mindspore::lite::RET_OK; }

  OpParameter *op_parameter() const { return op_parameter_; }

  bool InferShapeDone() const {
    if (std::any_of(in_tensors_.begin(), in_tensors_.end(),
                    [](lite::Tensor *input) { return input->data_type() == kObjectTypeTensorType; })) {
      return false;
    }
    auto shape = out_tensors_.front()->shape();
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      return false;
    }
    return true;
  }

  schema::PrimitiveType type() const override {
    return (this->op_parameter_ != nullptr) ? schema::PrimitiveType(this->op_parameter_->type_)
                                            : schema::PrimitiveType_NONE;
  }

  void set_inputs(const std::vector<mindspore::tensor::MSTensor *> &in_tensors) {
    this->in_tensors_.resize(in_tensors.size());
    (void)std::transform(in_tensors.begin(), in_tensors.end(), in_tensors_.begin(),
                         [](mindspore::tensor::MSTensor *tensor) { return static_cast<lite::Tensor *>(tensor); });
  }

  void set_outputs(const std::vector<mindspore::tensor::MSTensor *> &out_tensors) {
    this->out_tensors_.resize(out_tensors.size());
    (void)std::transform(out_tensors.begin(), out_tensors.end(), out_tensors_.begin(),
                         [](mindspore::tensor::MSTensor *tensor) { return static_cast<lite::Tensor *>(tensor); });
  }

  const std::vector<mindspore::MSTensor> &inputs() override {
    if (inputs_.empty()) {
      std::transform(in_tensors_.begin(), in_tensors_.end(), std::back_inserter(inputs_), [](lite::Tensor *tensor) {
        return mindspore::MSTensor(std::make_shared<mindspore::MSTensor::Impl>(tensor));
      });
    }
    return inputs_;
  }

  const std::vector<mindspore::MSTensor> &outputs() override {
    if (outputs_.empty()) {
      std::transform(out_tensors_.begin(), out_tensors_.end(), std::back_inserter(outputs_), [](lite::Tensor *tensor) {
        return mindspore::MSTensor(std::make_shared<mindspore::MSTensor::Impl>(tensor));
      });
    }
    return outputs_;
  }

  void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) { this->in_tensors_ = in_tensors; }

  virtual void set_in_tensor(lite::Tensor *in_tensor, size_t index) {
    if (index >= in_tensors_.size()) {
      MS_LOG(ERROR) << "index: " << index << " larger than in_tensors size: " << in_tensors_.size();
      return;
    }
    this->in_tensors_[index] = in_tensor;
  }

  void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) { this->out_tensors_ = out_tensors; }

  virtual void set_out_tensor(lite::Tensor *out_tensor, size_t index) {
    if (index >= out_tensors_.size()) {
      MS_LOG(ERROR) << "index: " << index << " larger than out_tensors size: " << out_tensors_.size();
      return;
    }
    this->out_tensors_[index] = out_tensor;
  }

  const std::vector<lite::Tensor *> &in_tensors() const { return in_tensors_; }

  const std::vector<lite::Tensor *> &out_tensors() const { return out_tensors_; }

  virtual int Train() {
    this->train_mode_ = true;
    return mindspore::lite::RET_OK;
  }

  virtual bool IsTrain() const { return this->train_mode_; }

  virtual int Eval() {
    this->train_mode_ = false;
    return mindspore::lite::RET_OK;
  }

  virtual bool IsEval() const { return !this->train_mode_; }

  virtual void SetTrainable(bool trainable = true) { this->trainable_ = trainable; }

  virtual bool IsTrainable() const { return this->trainable_; }

  TypeId registry_data_type(void) { return registry_data_type_; }

  void set_registry_data_type(TypeId data_type) { registry_data_type_ = data_type; }

  void set_workspace_size(size_t value) { workspace_size_ = value; }
  virtual size_t workspace_size() { return workspace_size_; }
  void AllocWorkspace();
  void FreeWorkspace();
  void *workspace() const { return workspace_; }
  void set_workspace(void *ws) {
    if (ws_allocated_ == false) {
      workspace_ = ws;
    }
  }
  const lite::Context *context() const { return this->ms_context_; }
  bool ws_allocated_ = false;

 protected:
  OpParameter *op_parameter_ = nullptr;
  // tensor will free in ~lite_session()
  std::vector<lite::Tensor *> in_tensors_;
  std::vector<lite::Tensor *> out_tensors_;
  bool train_mode_ = false;
  bool trainable_ = false;  // parameters of this Kernel are trained in Train Session
  TypeId registry_data_type_ = kTypeUnknown;
  size_t workspace_size_ = 0;
  void *workspace_ = nullptr;
  const lite::Context *ms_context_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_INNER_KERNEL_H_
