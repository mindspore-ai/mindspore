/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <vector>
#include "utils/log_adapter.h"
#include "ops/ops_func_impl/assign_add.h"

namespace mindspore::ops {
BaseShapePtr AssignAddFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto variable_shape_ptr = input_args[kIndex0]->GetShape();
  auto value_shape_ptr = input_args[kIndex1]->GetShape();
  if (variable_shape_ptr->IsDynamic() || value_shape_ptr->IsDynamic()) {
    return variable_shape_ptr->Clone();
  }

  auto variable_shape = variable_shape_ptr->GetShapeVector();
  auto value_shape = value_shape_ptr->GetShapeVector();
  if (variable_shape.size() != value_shape.size()) {
    if (variable_shape.size() == 1 && variable_shape[0] == 1 && value_shape.empty()) {
      return variable_shape_ptr->Clone();
    } else if (value_shape.size() == 1 && value_shape[0] == 1 && variable_shape.empty()) {
      return variable_shape_ptr->Clone();
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "','value' must have the same rank as 'variable'. But got 'value' rank: "
                               << value_shape.size() << ", 'variable' rank: " << variable_shape.size() << ".";
    }
  }

  for (uint64_t i = 0; i < variable_shape.size(); i++) {
    if (variable_shape[i] != value_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "','value' must have the same shape as 'variable'. But got 'value' shape: "
                               << value_shape_ptr->ToString()
                               << ", 'variable' shape: " << variable_shape_ptr->ToString() << ".";
    }
  }

  return variable_shape_ptr->Clone();
}

TypePtr AssignAddFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto variable_type_ptr = input_args[kIndex0]->GetType();
  return variable_type_ptr->Clone();
}
}  // namespace mindspore::ops
