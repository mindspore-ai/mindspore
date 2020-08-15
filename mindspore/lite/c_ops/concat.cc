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

#include "c_ops/concat.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Concat::GetAxis() const { return this->primitive->value.AsConcat()->axis; }
int Concat::GetN() const { return this->primitive->value.AsConcat()->n; }

void Concat::SetAxis(int axis) { this->primitive->value.AsConcat()->axis = axis; }
void Concat::SetN(int n) { this->primitive->value.AsConcat()->n = n; }

#else

int Concat::GetAxis() const { return this->primitive->value_as_Concat()->axis(); }
int Concat::GetN() const { return this->primitive->value_as_Concat()->n(); }

void Concat::SetAxis(int axis) {}
void Concat::SetN(int n) {}
#endif
namespace {
constexpr int kConcatOutputNum = 1;
}
int Concat::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  if (this->primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr!";
    return 1;
  }
  auto input0 = inputs_.front();
  auto output = outputs_.front();
  if (outputs_.size() != kConcatOutputNum) {
    MS_LOG(ERROR) << "output size is error";
    return 1;
  }
  MS_ASSERT(concat_prim != nullptr);
  auto input0_shape = inputs_.at(0)->shape();
  int axis = GetAxis() < 0 ? GetAxis() + input0_shape.size() : GetAxis();
  if (axis < 0 || axis >= input0_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis: " << axis;
    return 1;
  }

  auto input0_shape_without_axis = input0_shape;
  input0_shape_without_axis.erase(input0_shape_without_axis.begin() + axis);
  auto input0_data_type = inputs_.at(0)->data_type();
  schema::Format input0_format = inputs_[0]->GetFormat();
  int output_axis_dim = input0_shape.at(axis);
  for (size_t i = 1; i < inputs_.size(); ++i) {
    if (inputs_.at(i)->data_type() != input0_data_type) {
      MS_LOG(ERROR) << "All inputs should have the same data type!";
      return 1;
    }

    if (inputs_.at(i)->GetFormat() != input0_format) {
      MS_LOG(ERROR) << "All input format should be the same!";
      return 1;
    }
    auto shape_tmp = inputs_.at(i)->shape();
    if (shape_tmp.size() != input0_shape.size()) {
      MS_LOG(ERROR) << "All inputs should have the same dim num!";
      return 1;
    }
    auto axis_tmp = shape_tmp[axis];
    shape_tmp.erase(shape_tmp.begin() + axis);
    if (input0_shape_without_axis != shape_tmp) {
      MS_LOG(ERROR) << "Inputs should have the same dim except axis!";
      return 1;
    }
    output_axis_dim += axis_tmp;
  }
  auto output_shape = input0_shape;
  output_shape[axis] = output_axis_dim;
  outputs_[0]->set_shape(output_shape);
  output->set_data_type(input0->data_type());
  output->SetFormat(input0->GetFormat());

  return 0;
}
}  // namespace mindspore
