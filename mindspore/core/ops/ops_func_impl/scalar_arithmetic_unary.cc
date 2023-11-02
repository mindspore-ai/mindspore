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

#include "ops/ops_func_impl/scalar_arithmetic_unary.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScalarArithmeticUnaryFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  return abstract::kNoShape;
}

TypePtr ScalarArithmeticUnaryFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_type = input_args[0]->GetType();
  if (prim_name == kNameScalarLog) {
    return kFloat32;
  } else if (prim_name == kNameScalarBool) {
    return kBool;
  }
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
