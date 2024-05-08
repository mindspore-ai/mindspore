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

#include <set>
#include "ops/ops_func_impl/ceil.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr CeilFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  const int64_t max_dim = 8;
  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto in_shape = x_shape->GetShapeVector();
  (void)CheckAndConvertUtils::CheckInteger("The dimension of Ceil input", SizeToLong(in_shape.size()), kLessEqual,
                                           max_dim, prim_name);
  return x_shape->Clone();
}

TypePtr CeilFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  auto x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  // Valid types: kFloat16, kFloat32, kFloat64, kBFloat16.
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
  return x_type->Clone();
}
TypePtrList CeilFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input_x", input_type, valid_types, primitive->name());
  return {input_type};
}
ShapeArray CeilFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const int64_t max_dim = 8;
  auto in_shape = x_tensor->shape();
  (void)CheckAndConvertUtils::CheckInteger("The dimension of Ceil input", SizeToLong(in_shape.size()), kLessEqual,
                                           max_dim, primitive->name());
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameCeil, CeilFuncImpl)
}  // namespace ops
}  // namespace mindspore
