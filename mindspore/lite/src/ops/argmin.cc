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

#include "src/ops/argmin.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ArgMin::GetAxis() const { return this->primitive_->value.AsArgMin()->axis; }
bool ArgMin::GetOutMaxValue() const { return this->primitive_->value.AsArgMin()->outMaxValue; }
int ArgMin::GetTopK() const { return this->primitive_->value.AsArgMin()->topK; }
bool ArgMin::GetKeepDims() const { return this->primitive_->value.AsArgMin()->keepDims; }
int ArgMin::GetAxisType() const { return this->primitive_->value.AsArgMin()->axisType; }

void ArgMin::SetAxis(int axis) { this->primitive_->value.AsArgMin()->axis = axis; }
void ArgMin::SetOutMaxValue(bool out_max_value) { this->primitive_->value.AsArgMin()->outMaxValue = out_max_value; }
void ArgMin::SetTopK(int top_k) { this->primitive_->value.AsArgMin()->topK = top_k; }
void ArgMin::SetKeepDims(bool keep_dims) { this->primitive_->value.AsArgMin()->keepDims = keep_dims; }
void ArgMin::SetAxisType(int axis_type) { this->primitive_->value.AsArgMin()->axisType = axis_type; }

#else

int ArgMin::GetAxis() const { return this->primitive_->value_as_ArgMin()->axis(); }
bool ArgMin::GetOutMaxValue() const { return this->primitive_->value_as_ArgMin()->outMaxValue(); }
int ArgMin::GetTopK() const { return this->primitive_->value_as_ArgMin()->topK(); }
bool ArgMin::GetKeepDims() const { return this->primitive_->value_as_ArgMin()->keepDims(); }
int ArgMin::GetAxisType() const { return this->primitive_->value_as_ArgMin()->axisType(); }

#endif

int ArgMin::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "tensor number is error.";
  }
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape_size = input->shape().size();
  auto axis = GetAxis() < 0 ? GetAxis() + input_shape_size : GetAxis();
  if (axis >= input_shape_size || axis < 0) {
    MS_LOG(ERROR) << "Invalid axis " << GetAxis() << ", input shape size: " << input_shape_size;
    return RET_PARAM_INVALID;
  }
  std::vector<int> output_shape(input->shape());
  if (GetTopK() == 1 && !GetKeepDims()) {
    output_shape.erase(output_shape.begin() + axis);
  } else {
    output_shape[axis] = GetTopK();
  }

  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
