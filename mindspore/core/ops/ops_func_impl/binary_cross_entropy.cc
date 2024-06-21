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

#include "ops/ops_func_impl/binary_cross_entropy.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
TypePtr BinaryCrossEntropyFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type;
}

BaseShapePtr BinaryCrossEntropyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex3]);
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto reduction = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  if (reduction.has_value() && static_cast<Reduction>(reduction.value()) == Reduction::NONE) {
    return input_shape_ptr->Clone();
  }

  return std::make_shared<abstract::Shape>(ShapeVector{});
}
}  // namespace ops
}  // namespace mindspore
