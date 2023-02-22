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

#include "ops/tensor_array.h"

#include <vector>

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(TensorArray, BaseOperator);
constexpr auto kTensorArrayDynamicSize = "dynamic_size";
constexpr auto kTensorArrayIdenticalElementShapes = "identical_element_shapes";
constexpr auto kTensorArrayElementShape = "element_shape";
constexpr auto kTensorArrayDataType = "data_type";

void TensorArray::Init(bool dynamic_size, bool identical_element_shapes, const std::vector<int> &element_shape,
                       int data_type) {
  this->set_dynamic_size(dynamic_size);
  this->set_identical_element_shapes(identical_element_shapes);
  this->set_element_shape(element_shape);
  this->set_data_type(data_type);
}

void TensorArray::set_dynamic_size(bool dynamic_size) {
  (void)this->AddAttr(kTensorArrayDynamicSize, api::MakeValue(dynamic_size));
}

void TensorArray::set_identical_element_shapes(bool identical_element_shapes) {
  (void)this->AddAttr(kTensorArrayIdenticalElementShapes, api::MakeValue(identical_element_shapes));
}

void TensorArray::set_element_shape(const std::vector<int> &element_shape) {
  (void)this->AddAttr(kTensorArrayElementShape, api::MakeValue(element_shape));
}

void TensorArray::set_data_type(int data_type) { (void)this->AddAttr(kTensorArrayDataType, api::MakeValue(data_type)); }

bool TensorArray::get_dynamic_size() const {
  auto value_ptr = GetAttr(kTensorArrayDynamicSize);
  return GetValue<bool>(value_ptr);
}

bool TensorArray::get_identical_element_shapes() const {
  auto value_ptr = GetAttr(kTensorArrayIdenticalElementShapes);
  return GetValue<bool>(value_ptr);
}

const std::vector<int> TensorArray::get_element_shape() const {
  auto value_ptr = GetAttr(kTensorArrayElementShape);
  auto tmp = GetValue<std::vector<int64_t>>(value_ptr);
  std::vector<int> res(tmp.begin(), tmp.end());
  return res;
}

int TensorArray::get_data_type() const {
  auto value_ptr = GetAttr(kTensorArrayDataType);
  auto tmp = GetValue<int64_t>(value_ptr);
  return static_cast<int>(tmp);
}

REGISTER_PRIMITIVE_C(kNameTensorArray, TensorArray);
}  // namespace ops
}  // namespace mindspore
