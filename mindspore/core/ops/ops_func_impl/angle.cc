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

#include "ops/ops_func_impl/angle.h"
#include <vector>
#include <memory>
#include "ops/op_name.h"

namespace mindspore {
namespace ops {
BaseShapePtr AngleFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr AngleFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  auto input_tensor = input_type->cast<TensorTypePtr>();
  auto input_tensor_id = input_tensor->element()->type_id();
  // valid_types: kComplex64 and kComplex128
  return input_tensor_id == kNumberTypeComplex64 ? std::make_shared<TensorType>(kFloat32)
                                                 : std::make_shared<TensorType>(kFloat64);
}
}  // namespace ops
}  // namespace mindspore
