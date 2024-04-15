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

#include "ops/ops_func_impl/norm.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr NormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return NormInferShape(primitive, input_args);
}

TypePtr NormFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args[kInputIndex4]->GetType()->isa<TypeNone>()) {
    return input_args[0]->GetType()->Clone();
  }
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
  if (!dtype_ptr.has_value()) {
    return input_args[0]->GetType()->Clone();
  }
  return std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_ptr.value())));
}
}  // namespace ops
}  // namespace mindspore
