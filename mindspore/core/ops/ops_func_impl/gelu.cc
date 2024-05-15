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
#include "ops/ops_func_impl/gelu.h"
#include <set>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GeLUFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);

  return x_shape->Clone();
}

TypePtr GeLUFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kBFloat16, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
  MS_EXCEPTION_IF_NULL(x_type);
  return x_type->Clone();
}
TypePtrList GeLUFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_type = x_tensor->Dtype();
  const std::set<TypePtr> valid_types = {kBFloat16, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, valid_types, primitive->name());
  return {x_type};
}
ShapeArray GeLUFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameGeLU, GeLUFuncImpl)
}  // namespace ops
}  // namespace mindspore
