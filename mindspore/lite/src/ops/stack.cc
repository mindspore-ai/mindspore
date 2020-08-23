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

#include "src/ops/stack.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Stack::GetAxis() const { return this->primitive_->value.AsStack()->axis; }
int Stack::GetN() const { return this->primitive_->value.AsStack()->n; }
std::vector<int> Stack::GetIsScale() const { return this->primitive_->value.AsStack()->isScale; }

void Stack::SetAxis(int axis) { this->primitive_->value.AsStack()->axis = axis; }
void Stack::SetN(int n) { this->primitive_->value.AsStack()->n = n; }
void Stack::SetIsScale(const std::vector<int> &is_scale) { this->primitive_->value.AsStack()->isScale = is_scale; }

#else

int Stack::GetAxis() const { return this->primitive_->value_as_Stack()->axis(); }
int Stack::GetN() const { return this->primitive_->value_as_Stack()->n(); }
std::vector<int> Stack::GetIsScale() const {
  auto fb_vector = this->primitive_->value_as_Stack()->isScale();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void Stack::SetAxis(int axis) {}
void Stack::SetN(int n) {}
void Stack::SetIsScale(const std::vector<int> &is_scale) {}
#endif

namespace {
constexpr int kStackOutputNum = 1;
constexpr int kStackMinInputNum = 2;
}  // namespace
int Stack::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (outputs.size() != kStackOutputNum) {
    MS_LOG(ERROR) << "Invalid output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  if (inputs.size() < kStackMinInputNum) {
    MS_LOG(ERROR) << "Invalid input size " << inputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  auto input0_data_type = input->data_type();
  outputs[0]->set_data_type(input0_data_type);
  outputs[0]->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape = input->shape();

  std::vector<int32_t> output_shape = input_shape;
  auto axis = GetAxis() < 0 ? GetAxis() + input_shape.size() : GetAxis();
  if (axis < 0 || axis > input_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis " << GetAxis();
    return RET_PARAM_INVALID;
  }

  for (size_t i = 1; i < inputs.size(); ++i) {
    auto input_shape_tmp = inputs[i]->shape();
    if (input_shape_tmp.size() != input_shape.size()) {
      MS_LOG(ERROR) << "All input shape size should be the same!";
      return RET_PARAM_INVALID;
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (input_shape_tmp[j] != input_shape[j]) {
        MS_LOG(ERROR) << "All input shape should be the same!";
        return RET_PARAM_INVALID;
      }
    }
    if (inputs[i]->data_type() != input0_data_type) {
      MS_LOG(ERROR) << "All input shuld have the same data type!input[" << i
                    << "] data type = " << inputs[i]->data_type();
      return RET_PARAM_INVALID;
    }
  }
  output_shape.insert(output_shape.begin() + axis, inputs.size());
  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
