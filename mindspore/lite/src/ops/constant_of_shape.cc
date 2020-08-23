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

#include "src/ops/constant_of_shape.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kShapeInputNum = 1;
constexpr int kShapeOutputNum = 1;
}  // namespace
#ifdef PRIMITIVE_WRITEABLE
float ConstantOfShape::GetValue() const { return this->primitive_->value.AsConstantOfShape()->value; }

void ConstantOfShape::SetValue(float value) { this->primitive_->value.AsConstantOfShape()->value = value; }

#else

float ConstantOfShape::GetValue() const { return this->primitive_->value_as_ConstantOfShape()->value(); }

#endif

int ConstantOfShape::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != kShapeInputNum) {
    MS_LOG(ERROR) << "inputs to ConstantOfShape operator should be 1, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  if (inputs_.front() == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr!";
    return RET_PARAM_INVALID;
  }
  if (outputs_.size() != kShapeOutputNum) {
    MS_LOG(ERROR) << "outputs to ConstantOfShape operator should be 1, but " << outputs_.size() << " is given.";
    return RET_ERROR;
  }
  auto in_tensor = inputs_.front();
  auto out_tensor = outputs_.front();
  out_tensor->set_data_type(kNumberTypeFloat32);
  out_tensor->SetFormat(in_tensor->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto in_data = reinterpret_cast<int *>(in_tensor->Data());
  int size = in_tensor->ElementsNum();
  std::vector<int> out_shape(size);
  for (int i = 0; i < size; ++i) {
    out_shape[i] = in_data[i];
  }
  out_tensor->set_shape(out_shape);

  return RET_OK;
}
}  // namespace mindspore::lite
