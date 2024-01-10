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

#include <vector>
#include "ops/ops_func_impl/sin.h"

namespace mindspore::ops {
BaseShapePtr SinFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr SinFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  auto input_type_id = input_type->cast<TensorTypePtr>()->element()->type_id();
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8, kNumberTypeInt8,  kNumberTypeInt16,
                                                  kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_int_or_bool) {
    return std::make_shared<TensorType>(kFloat32);
  } else {
    return input_type->Clone();
  }
}
}  // namespace mindspore::ops
