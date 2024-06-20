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

#include "ops/ops_func_impl/binary_cross_entropy_with_logits_backward.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
TypePtr BinaryCrossEntropyWithLogitsBackwardFuncImpl::InferType(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto target_type = input_args[kInputIndex2]->GetType();
  return target_type;
}

BaseShapePtr BinaryCrossEntropyWithLogitsBackwardFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kInputIndex1]->GetShape();

  return input_shape_ptr->Clone();
}

ShapeArray BinaryCrossEntropyWithLogitsBackwardFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                                    const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape_vector = x_tensor->shape();
  return {x_shape_vector};
}

TypePtrList BinaryCrossEntropyWithLogitsBackwardFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                    const ValuePtrList &input_values) const {
  const auto &target_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  return {target_tensor->Dtype()};
}
REGISTER_SIMPLE_INFER(kNameBinaryCrossEntropyWithLogitsBackward, BinaryCrossEntropyWithLogitsBackwardFuncImpl)
}  // namespace ops
}  // namespace mindspore
