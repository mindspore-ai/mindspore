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

#include "ops/ops_func_impl/binary_cross_entropy_with_logits.h"
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
TypePtr BCEWithLogitsLossFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto target_type = input_args[kInputIndex1]->GetType();
  auto input_type = input_args[kInputIndex0]->GetType();
  auto weight_type = input_args[kInputIndex2]->GetType();
  auto pos_weight_type = input_args[kInputIndex3]->GetType();
  std::set<TypePtr> valid_types = {kFloat32, kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTypeValid("target", target_type, valid_types, op_name);
  if (weight_type->type_id() != kMetaTypeNone) {
    (void)CheckAndConvertUtils::CheckTypeValid("weight", weight_type, valid_types, op_name);
  }
  if (pos_weight_type->type_id() != kMetaTypeNone) {
    (void)CheckAndConvertUtils::CheckTypeValid("pos_weight", pos_weight_type, valid_types, op_name);
  }
  return target_type;
}

BaseShapePtr BCEWithLogitsLossFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto input_shape_vector = input_shape_ptr->GetShapeVector();

  auto target_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto target_shape_vector = target_shape_ptr->GetShapeVector();
  if (!ObscureShapeEqual(input_shape_vector, target_shape_vector)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', two inputs 'input' and 'target' shape are not equal, 'input' shape: "
                             << input_shape_vector << ", 'target' shape: " << target_shape_vector;
  }

  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(weight_shape_ptr);
  if (!weight_shape_ptr->isa<abstract::NoShape>()) {
    auto weight_shape_vector = weight_shape_ptr->GetShapeVector();
    if (!IsBroadcastable(input_shape_vector, weight_shape_vector)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the shape of 'weight' can not broadcast to the shape of 'input'.";
    }
  }

  auto pos_weight_shape_ptr = input_args[kInputIndex3]->GetShape();
  MS_EXCEPTION_IF_NULL(pos_weight_shape_ptr);
  if (!pos_weight_shape_ptr->isa<abstract::NoShape>()) {
    auto pos_weight_shape_vector = pos_weight_shape_ptr->GetShapeVector();
    if (!IsBroadcastable(input_shape_vector, pos_weight_shape_vector)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the shape of 'pos_weight' can not broadcast to the shape of 'input'.";
    }
  }

  auto reduction = GetScalarValue<int64_t>(input_args[kInputIndex4]->BuildValue());
  if (reduction.has_value() && static_cast<Reduction>(reduction.value()) == Reduction::NONE) {
    return input_shape_ptr->Clone();
  }
  return std::make_shared<abstract::Shape>(ShapeVector{});
}

ShapeArray BCEWithLogitsLossFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape_vector = x_tensor->shape();

  const auto &target_tensor = input_values[kIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  auto target_shape_vector = target_tensor->shape();
  if (x_shape_vector != target_shape_vector) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', two inputs 'input' and 'target' shape are not equal, 'input' shape: "
                             << x_shape_vector << ", 'target' shape: " << target_shape_vector;
  }

  if (input_values[kInputIndex2] != mindspore::kNone) {
    const auto &weight_tensor = input_values[kIndex2]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(weight_tensor);
    auto weight_shape_vector = weight_tensor->shape();
    if (!IsBroadcastable(x_shape_vector, weight_shape_vector)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the shape of 'weight' can not broadcast to the shape of 'input'.";
    }
  }

  if (input_values[kInputIndex3] != mindspore::kNone) {
    const auto &pos_weight_tensor = input_values[kInputIndex3]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(pos_weight_tensor);
    auto pos_weight_shape_vector = pos_weight_tensor->shape();
    if (!IsBroadcastable(x_shape_vector, pos_weight_shape_vector)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the shape of 'pos_weight' can not broadcast to the shape of 'input'.";
    }
  }

  const auto &reduction = input_values[kIndex4]->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(reduction);
  auto reduction_value = reduction->value();
  if (static_cast<Reduction>(reduction_value) == Reduction::NONE) {
    return {x_shape_vector};
  }

  return {ShapeVector{}};
}

TypePtrList BCEWithLogitsLossFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const ValuePtrList &input_values) const {
  auto op_name = primitive->name();
  std::set<TypePtr> valid_types = {kFloat32, kFloat16, kBFloat16};
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_tensor->Dtype(), valid_types, op_name);
  const auto &target_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  (void)CheckAndConvertUtils::CheckTypeValid("target", target_tensor->Dtype(), valid_types, op_name);
  if (input_values[kInputIndex2] != mindspore::kNone) {
    const auto &weight_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(weight_tensor);
    (void)CheckAndConvertUtils::CheckTypeValid("weight", weight_tensor->Dtype(), valid_types, op_name);
  }
  if (input_values[kInputIndex3] != mindspore::kNone) {
    const auto &pos_weight_tensor = input_values[kInputIndex3]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(pos_weight_tensor);
    (void)CheckAndConvertUtils::CheckTypeValid("pos_weight", pos_weight_tensor->Dtype(), valid_types, op_name);
  }

  return {target_tensor->Dtype()};
}
REGISTER_SIMPLE_INFER(kNameBCEWithLogitsLoss, BCEWithLogitsLossFuncImpl)
}  // namespace ops
}  // namespace mindspore
