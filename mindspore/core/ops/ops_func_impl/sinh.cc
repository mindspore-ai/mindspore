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
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/sinh.h"

namespace mindspore::ops {
BaseShapePtr SinhFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  const auto input_dim = input_shape->GetShapeVector().size();
  const size_t max_dim = 9;
  MS_CHECK_VALUE(input_dim < max_dim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of x", input_dim, kLessThan, max_dim, primitive));
  return input_shape->Clone();
}

TypePtr SinhFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace mindspore::ops
