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

#include "ops/tensor_list_stack.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(TensorListStack, BaseOperator);
void TensorListStack::Init(const int64_t num_elements, const int64_t element_dtype) {
  this->set_num_elements(num_elements);
  this->set_element_dtype(element_dtype);
}

void TensorListStack::set_num_elements(const int64_t num_elements) {
  (void)this->AddAttr(kNumElements, api::MakeValue(num_elements));
}

void TensorListStack::set_element_dtype(const int64_t element_dtype) {
  (void)this->AddAttr(kElement_dtype, api::MakeValue(element_dtype));
}

int64_t TensorListStack::get_num_elements() const {
  auto value_ptr = GetAttr(kNumElements);
  return GetValue<int64_t>(value_ptr);
}

int64_t TensorListStack::get_element_dtype() const {
  auto value_ptr = GetAttr(kElement_dtype);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameTensorListStack, TensorListStack);
}  // namespace ops
}  // namespace mindspore
