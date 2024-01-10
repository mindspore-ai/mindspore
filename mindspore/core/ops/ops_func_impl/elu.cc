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

#include "ops/ops_func_impl/elu.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EluFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto alpha_ptr = GetScalarValue<float>(input_args[1]->GetValue());
  if (alpha_ptr.has_value()) {
    auto alpha = alpha_ptr.value();
    MS_CHECK_VALUE(alpha == 1.0, primitive->name() + "In Elu, alpha must be 1.0f.");
  }
  return input_args[0]->GetShape()->Clone();
}

TypePtr EluFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
