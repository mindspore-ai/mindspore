/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/constant_of_shape.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void ConstantOfShape::Init(int64_t data_type, const std::vector<float> &value) {
  this->set_data_type(data_type);
  this->set_value(value);
}

void ConstantOfShape::set_data_type(int64_t data_type) { (void)this->AddAttr(kDataType, api::MakeValue(data_type)); }

int64_t ConstantOfShape::get_data_type() const {
  auto value_ptr = this->GetAttr(kDataType);
  return GetValue<int64_t>(value_ptr);
}

void ConstantOfShape::set_value(const std::vector<float> &value) { (void)this->AddAttr(kValue, api::MakeValue(value)); }

std::vector<float> ConstantOfShape::get_value() const {
  auto value_ptr = this->GetAttr(kValue);
  return GetValue<std::vector<float>>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ConstantOfShape, BaseOperator);
REGISTER_PRIMITIVE_C(kNameConstantOfShape, ConstantOfShape);
}  // namespace ops
}  // namespace mindspore
