/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/reduce.h"
#include <memory>

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Reduce::GetAxes() const { return this->primitive_->value.AsReduce()->axes; }
int Reduce::GetKeepDims() const { return this->primitive_->value.AsReduce()->keepDims; }
int Reduce::GetMode() const { return this->primitive_->value.AsReduce()->mode; }

void Reduce::SetAxes(const std::vector<int> &axes) { this->primitive_->value.AsReduce()->axes = axes; }
void Reduce::SetKeepDims(int keep_dims) { this->primitive_->value.AsReduce()->keepDims = keep_dims; }
void Reduce::SetMode(int mode) { this->primitive_->value.AsReduce()->mode = (schema::ReduceMode)mode; }

int Reduce::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  this->primitive_ = new (schema::PrimitiveT);
  auto attr = std::make_unique<schema::ReduceT>();
  attr->mode = schema::ReduceMode_ReduceMean;

  attr->keepDims = GetValue<bool>(prim.GetAttr("keep_dims"));
  if (inputs.size() == kAnfPopulaterTwo) {
    auto inputNode = inputs[kAnfPopulaterOne];
    MS_ASSERT(inputNode != nullptr);
    if (inputNode->isa<ValueNode>()) {
      auto valueNode = inputNode->cast<ValueNodePtr>();
      MS_ASSERT(valueNode != nullptr);
      auto value = valueNode->value();
      MS_ASSERT(value != nullptr);
      if (value->isa<ValueTuple>()) {
        auto valTuplPtr = dyn_cast<ValueTuple>(value);
        MS_ASSERT(valTuplPtr != nullptr);
        for (size_t i = 0; i < valTuplPtr->size(); i++) {
          auto elem = dyn_cast<Int32Imm>((*valTuplPtr)[i]);
          MS_ASSERT(elem != nullptr);
          attr->axes.emplace_back(elem->value());
        }
      }
    }
  }

  this->primitive_->value.type = schema::PrimitiveType_Reduce;
  this->primitive_->value.value = attr.release();

  return RET_OK;
}

#else

std::vector<int> Reduce::GetAxes() const {
  auto fb_vector = this->primitive_->value_as_Reduce()->axes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Reduce::GetKeepDims() const { return this->primitive_->value_as_Reduce()->keepDims(); }
int Reduce::GetMode() const { return this->primitive_->value_as_Reduce()->mode(); }

void Reduce::SetAxes(const std::vector<int> &axes) {}
void Reduce::SetKeepDims(int keep_dims) {}
void Reduce::SetMode(int mode) {}
#endif

namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
}  // namespace
int Reduce::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != kInputSize || outputs_.size() != kOutputSize) {
    return RET_ERROR;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  bool keep_dims = static_cast<bool>(GetKeepDims());
  std::vector<int> in_shape = input->shape();
  std::vector<int> out_shape;
  const auto &axes = GetAxes();
  auto num_axes = axes.size();
  // reduce on all axes
  if (num_axes == 0) {
    if (keep_dims) {
      for (size_t i = 0; i < in_shape.size(); i++) {
        out_shape.push_back(1);
      }
    }
    output->set_shape(out_shape);
    output->set_data_type(input->data_type());
    return RET_OK;
  }
  // reduce on selected axes
  for (size_t i = 0; i < in_shape.size(); i++) {
    bool reduce_axis = false;
    for (size_t idx = 0; idx < num_axes; ++idx) {
      if (static_cast<size_t>(axes[idx]) == i || static_cast<size_t>(axes[idx] + in_shape.size()) == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
