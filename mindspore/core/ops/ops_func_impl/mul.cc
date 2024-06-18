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

#include <map>
#include <string>
#include "ops/ops_func_impl/mul.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
BaseShapePtr MulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr MulFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]->GetType());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());
  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types_with_complex_and_bool,
                                                           primitive->name());
}

TypePtrList MulFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();
  return {input_type};
}
ShapeArray MulFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {BroadCastInferShape(primitive->name(), input_values)};
}
REGISTER_SIMPLE_INFER(kNameMul, MulFuncImpl)
}  // namespace mindspore::ops
