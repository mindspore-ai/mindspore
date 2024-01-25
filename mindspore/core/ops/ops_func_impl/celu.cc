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
#include "ops/ops_func_impl/celu.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CeLUFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, 2, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  // check attr: alpha
  auto input_alpha = input_args[kIndex1];
  MS_EXCEPTION_IF_NULL(input_alpha);
  ValuePtr alpha_ptr = input_alpha->GetValue();
  auto alpha_value = GetScalarValue<float>(alpha_ptr);
  if (MS_LIKELY(alpha_value.has_value())) {
    (void)CheckAndConvertUtils::CheckValue<float>(kAlpha, alpha_value.value(), kNotEqual, 0.0f, prim_name);
  }

  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  return x_shape->Clone();
}

TypePtr CeLUFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("CeLU input numbers", SizeToLong(input_args.size()), kEqual, 2, prim_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
