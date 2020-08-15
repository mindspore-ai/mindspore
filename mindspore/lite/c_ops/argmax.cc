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

#include "c_ops/argmax.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int ArgMax::GetAxis() const { return this->primitive->value.AsArgMax()->axis; }
bool ArgMax::GetOutMaxValue() const { return this->primitive->value.AsArgMax()->outMaxValue; }
int ArgMax::GetTopK() const { return this->primitive->value.AsArgMax()->topK; }
bool ArgMax::GetKeepDims() const { return this->primitive->value.AsArgMax()->keepDims; }
int ArgMax::GetAxisType() const { return this->primitive->value.AsArgMax()->axisType; }

void ArgMax::SetAxis(int axis) { this->primitive->value.AsArgMax()->axis = axis; }
void ArgMax::SetOutMaxValue(bool out_max_value) { this->primitive->value.AsArgMax()->outMaxValue = out_max_value; }
void ArgMax::SetTopK(int top_k) { this->primitive->value.AsArgMax()->topK = top_k; }
void ArgMax::SetKeepDims(bool keep_dims) { this->primitive->value.AsArgMax()->keepDims = keep_dims; }
void ArgMax::SetAxisType(int axis_type) { this->primitive->value.AsArgMax()->axisType = axis_type; }

#else

int ArgMax::GetAxis() const { return this->primitive->value_as_ArgMax()->axis(); }
bool ArgMax::GetOutMaxValue() const { return this->primitive->value_as_ArgMax()->outMaxValue(); }
int ArgMax::GetTopK() const { return this->primitive->value_as_ArgMax()->topK(); }
bool ArgMax::GetKeepDims() const { return this->primitive->value_as_ArgMax()->keepDims(); }
int ArgMax::GetAxisType() const { return this->primitive->value_as_ArgMax()->axisType(); }

void ArgMax::SetAxis(int axis) {}
void ArgMax::SetOutMaxValue(bool out_max_value) {}
void ArgMax::SetTopK(int top_k) {}
void ArgMax::SetKeepDims(bool keep_dims) {}
void ArgMax::SetAxisType(int axis_type) {}
#endif
int ArgMax::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "tensor number is error.";
  }

  std::vector<int> output_shape(input->shape());
  auto input_shape_size = input->shape().size();
  int axis = GetAxis() < 0 ? GetAxis() + input_shape_size : GetAxis();
  if (axis >= input_shape_size || axis < 0) {
    MS_LOG(ERROR) << "Invalid axis " << GetAxis() << ", input shape size: " << input_shape_size;
    return 1;
  }
  if (GetTopK() == 1 && !GetKeepDims()) {
    output_shape.erase(output_shape.begin() + axis);
  } else {
    output_shape[axis] = GetTopK();
  }

  output->SetFormat(input->GetFormat());
  output->set_shape(output_shape);
  output->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
