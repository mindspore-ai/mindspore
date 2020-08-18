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

#include "src/ops/one_hot.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int OneHot::GetAxis() const { return this->primitive->value.AsOneHot()->axis; }

void OneHot::SetAxis(int axis) { this->primitive->value.AsOneHot()->axis = axis; }

#else

int OneHot::GetAxis() const { return this->primitive->value_as_OneHot()->axis(); }

void OneHot::SetAxis(int axis) {}
#endif

namespace {
constexpr size_t kOneHotInputNum = 4;
}
int OneHot::InferShape(std::vector<tensor::Tensor *> inputs, std::vector<tensor::Tensor *> outputs) {
  if (this->primitive == nullptr) {
    return RET_NULL_PTR;
  }

  int axis = GetAxis();
  // indices, depth, on_value, off_value
  if (inputs.size() != kOneHotInputNum) {
    MS_LOG(ERROR) << "OneHot got inputs num " << inputs.size() << ", should be " << kOneHotInputNum;
    return RET_ERROR;
  }
  auto depth_tensor = inputs.at(1);
  if (depth_tensor == nullptr) {
    return RET_NULL_PTR;
  }
  const int *depth = static_cast<int *>(depth_tensor->Data());
  auto input = inputs.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto on_value = inputs.at(2);
  if (on_value == nullptr) {
    return RET_NULL_PTR;
  }
  auto output = outputs.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(on_value->data_type());
  output->SetFormat(on_value->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  const auto input_shape = input->shape();
  int input_rank = static_cast<int>(input_shape.size());
  if (axis < 0) {
    axis += input_rank + 1;
  }
  std::vector<int> output_shape(input_shape);
  output_shape.insert(output_shape.cbegin() + axis, *depth);
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
