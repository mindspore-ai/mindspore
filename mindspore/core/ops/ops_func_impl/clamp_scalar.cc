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

#include "ops/ops_func_impl/clamp_scalar.h"
#include <vector>
#include <map>
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
TypePtr ClampScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input0_type = input_args[kInputIndex0]->GetType();
  auto input1_type = input_args[kInputIndex1]->GetType();
  auto input2_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(input0_type);
  MS_EXCEPTION_IF_NULL(input1_type);
  MS_EXCEPTION_IF_NULL(input2_type);
  if (input1_type->type_id() == kMetaTypeNone && input2_type->type_id() == kMetaTypeNone) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }

  if (input0_type->type_id() == kNumberTypeBool || input1_type->type_id() == kNumberTypeBool ||
      input2_type->type_id() == kNumberTypeBool) {
    MS_EXCEPTION(ValueError) << "For Clamp, the dtype of 'input', 'min' and 'max' must not be bool.";
  }

  return input0_type->Clone();
}

BaseShapePtr ClampScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input0_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input0_shape);
  return input0_shape->Clone();
}
}  // namespace mindspore::ops
