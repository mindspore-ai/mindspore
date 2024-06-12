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

#include <map>
#include <set>
#include <string>
#include "ops/ops_func_impl/isfinite.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr IsFiniteFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr IsFiniteFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  static const std::set<TypePtr> number_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,   kUInt16,
                                                 kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBFloat16};

  auto input_type = input_args[kInputIndex0]->GetType();
  const auto &prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, number_types, prim_name);
  return std::make_shared<TensorType>(kBool);
}
TypePtrList IsFiniteFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();
  static const std::set<TypePtr> number_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,   kUInt16,
                                                 kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("x", input_type, number_types, primitive->name());
  return {kBool};
}
ShapeArray IsFiniteFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameIsFinite, IsFiniteFuncImpl)
}  // namespace ops
}  // namespace mindspore
