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

#include "ops/ops_func_impl/copy.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CopyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  return x_shape->Clone();
}

TypePtr CopyFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
