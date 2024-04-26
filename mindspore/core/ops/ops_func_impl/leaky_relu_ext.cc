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
#include "ops/ops_func_impl/leaky_relu_ext.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LeakyReLUExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto input_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input_shape);
  return input_shape->Clone();
}

TypePtr LeakyReLUExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  static const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kBFloat16};
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto input_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_type, valid_types, prim_name);
  return input_type;
}
}  // namespace ops
}  // namespace mindspore
