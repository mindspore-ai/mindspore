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

#include "ops/ops_func_impl/xlogy_scalar_self.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr XLogYScalarSelfFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input1_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input1_shape);
  return input1_shape->Clone();
}

TypePtr XLogYScalarSelfFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x1_type = input_args[kInputIndex0]->GetType();
  auto x2_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x1_type);
  MS_EXCEPTION_IF_NULL(x2_type);
  return PromoteType(x1_type, x2_type, primitive->name());
}
}  // namespace ops
}  // namespace mindspore
