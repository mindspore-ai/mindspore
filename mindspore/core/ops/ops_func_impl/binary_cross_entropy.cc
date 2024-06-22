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
#include "ops/ops_func_impl/simple_infer.h"

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
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto target_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  if (!input_shape_ptr->isa<abstract::NoShape>() && !target_shape_ptr->isa<abstract::NoShape>() &&
      !weight_shape_ptr->isa<abstract::NoShape>()) {
    auto &input_shape = input_shape_ptr->GetShapeVector();
    auto &target_shape = target_shape_ptr->GetShapeVector();
    auto &weight_shape = weight_shape_ptr->GetShapeVector();
    MS_CHECK_VALUE(input_shape == target_shape,
                   CheckAndConvertUtils::FormatCheckMsg("input_shape", input_shape, kEqual, target_shape, primitive));
    MS_CHECK_VALUE(input_shape == weight_shape,
                   CheckAndConvertUtils::FormatCheckMsg("input_shape", input_shape, kEqual, weight_shape, primitive));
  }
  auto reduction = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());

  if (reduction.has_value() && static_cast<Reduction>(reduction.value()) == Reduction::NONE) {
    return input_shape_ptr->Clone();
  }

  return std::make_shared<abstract::Shape>(ShapeVector{});
}

TypePtrList BinaryCrossEntropyFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype()};
}

ShapeArray BinaryCrossEntropyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto &input_shape = input_tensor->shape();
  const auto &target_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  auto target_shape = target_tensor->shape();
  const auto &weight_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(weight_tensor);
  auto &weight_shape = weight_tensor->shape();
  MS_CHECK_VALUE(input_shape == target_shape,
                 CheckAndConvertUtils::FormatCheckMsg("input_shape", input_shape, kEqual, target_shape, primitive));
  MS_CHECK_VALUE(input_shape == weight_shape,
                 CheckAndConvertUtils::FormatCheckMsg("input_shape", input_shape, kEqual, weight_shape, primitive));
  const auto &reduction = input_values[kInputIndex3]->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(reduction);
  auto reduction_value = reduction->value();
  if (static_cast<Reduction>(reduction_value) == Reduction::NONE) {
    return {input_shape};
  }

  return {ShapeVector{}};
}

REGISTER_SIMPLE_INFER(kNameBinaryCrossEntropy, BinaryCrossEntropyFuncImpl)

}  // namespace ops
}  // namespace mindspore
