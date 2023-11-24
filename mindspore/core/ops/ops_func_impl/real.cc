/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/real.h"
#include <memory>

namespace mindspore {
namespace ops {
BaseShapePtr RealFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  return x_shape->Clone();
}

TypePtr RealFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(x_type->cast<TensorTypePtr>());
  MS_EXCEPTION_IF_NULL(x_type->cast<TensorTypePtr>()->element());

  auto x_type_id = x_type->cast<TensorTypePtr>()->element()->type_id();
  if (x_type_id == kNumberTypeComplex64) {
    return std::make_shared<TensorType>(kFloat32);
  }
  if (x_type_id == kNumberTypeComplex128) {
    return std::make_shared<TensorType>(kFloat64);
  }
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
