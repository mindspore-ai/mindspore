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

#include "ops/ops_func_impl/sigmoid.h"

#include <vector>
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr SigmoidFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex0]->GetShape();
  return x_shape->Clone();
}

TypePtr SigmoidFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "Input for Sigmoid should be TensorType, but got " << TypeIdToString(x_type->type_id());
  }
  auto x_type_id = x_type->cast<TensorTypePtr>()->element()->type_id();
  const std::set<TypeId> int_or_bool = {kNumberTypeUInt8, kNumberTypeInt8,  kNumberTypeInt16,
                                        kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&x_type_id](const TypeId &type_id) { return x_type_id == type_id; });
  if (is_int_or_bool) {
    return std::make_shared<TensorType>(kFloat32);
  } else {
    return x_type->Clone();
  }
}

}  // namespace ops
}  // namespace mindspore
