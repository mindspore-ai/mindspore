/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/right_shift.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr RightShiftFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr RightShiftFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto input_x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_x_type);
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  auto input_y_type = input_args[kIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_y_type);
  if (*input_x_type != *input_y_type) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the dtype of two args should be same, but the first arg dtype "
                            << input_x_type->ToString() << " are not consistent with second arg dtype "
                            << input_y_type->ToString();
  }
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, valid_types, prim_name);
  return input_x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
