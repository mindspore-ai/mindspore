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

#ifndef MINDSPORE_LITE_SRC_KERNEL_H_
#define MINDSPORE_LITE_SRC_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include "schema/model_generated.h"
#include "include/lite_utils.h"
#include "include/context.h"

namespace mindspore::kernel {
class Kernel {
 public:
  Kernel() = default;

  Kernel(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
         const schema::Primitive *primitive, const lite::Context *ctx)
      : inputs_(std::move(inputs)), outputs_(std::move(outputs)), primitive_(primitive), context_(ctx) {
    if (primitive != nullptr) {
      type_ = primitive->value_type();
    }
  }

  virtual ~Kernel() = default;

  virtual int Prepare() = 0;

  virtual int Execute() = 0;

  virtual int ReSize() = 0;

  virtual schema::PrimitiveType type() const { return type_; }

  virtual void set_inputs(const std::vector<mindspore::tensor::MSTensor *> &in_tensors) { this->inputs_ = in_tensors; }
  virtual void set_input(mindspore::tensor::MSTensor *in_tensor, int index) { this->inputs_[index] = in_tensor; }

  virtual void set_outputs(const std::vector<mindspore::tensor::MSTensor *> &out_tensors) {
    this->outputs_ = out_tensors;
  }

  virtual void set_output(mindspore::tensor::MSTensor *out_tensor, int index) { this->outputs_[index] = out_tensor; }

  virtual const std::vector<mindspore::tensor::MSTensor *> &inputs() { return this->inputs_; }

  virtual const std::vector<mindspore::tensor::MSTensor *> &outputs() { return this->outputs_; }

  std::string name() const { return this->name_; }

  void set_name(const std::string &name) { this->name_ = name; }
  const lite::Context *context() const { return this->context_; }
  const schema::Primitive *primitive() const { return this->primitive_; }

 protected:
  std::vector<mindspore::tensor::MSTensor *> inputs_;
  std::vector<mindspore::tensor::MSTensor *> outputs_;
  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
  std::string name_;
  const schema::Primitive *primitive_ = nullptr;
  const lite::Context *context_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_KERNEL_H_
