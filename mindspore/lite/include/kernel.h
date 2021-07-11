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

#ifndef MINDSPORE_INCLUDE_API_KERNEL_H
#define MINDSPORE_INCLUDE_API_KERNEL_H
#include <vector>
#include <string>
#include <utility>
#include "schema/model_generated.h"
#include "include/api/types.h"
#include "include/api/context.h"

namespace mindspore::kernel {
class Kernel {
 public:
  Kernel() = default;

  Kernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
         const schema::Primitive *primitive, const mindspore::Context *ctx)
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

  virtual void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors) { this->inputs_ = in_tensors; }

  virtual void set_input(mindspore::MSTensor in_tensor, int index) { this->inputs_[index] = in_tensor; }

  virtual void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors) { this->outputs_ = out_tensors; }

  virtual void set_output(mindspore::MSTensor out_tensor, int index) { this->outputs_[index] = out_tensor; }

  virtual const std::vector<mindspore::MSTensor> &inputs() { return this->inputs_; }

  virtual const std::vector<mindspore::MSTensor> &outputs() { return this->outputs_; }

  std::string name() const { return this->name_; }

  void set_name(const std::string &name) { this->name_ = name; }

  const mindspore::Context *context() const { return this->context_; }

  const schema::Primitive *primitive() const { return this->primitive_; }

 protected:
  std::vector<mindspore::MSTensor> inputs_;
  std::vector<mindspore::MSTensor> outputs_;
  schema::PrimitiveType type_ = schema::PrimitiveType_NONE;
  std::string name_;
  const schema::Primitive *primitive_ = nullptr;
  const mindspore::Context *context_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_INCLUDE_API_KERNEL_H
