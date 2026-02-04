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

#include "infer/ops_func_impl/binary_cross_entropy.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {
TypePtr BinaryCrossEntropyFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_elememt = input_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_elememt);
  auto input_type_id = input_elememt->type_id();
  if (valid_types_.find(input_type_id) == valid_types_.end()) {
    MS_EXCEPTION(TypeError) << "For Primitive[BinaryCrossEntorpy], the type of the input must be "
                               "[Float16, Float32, BFloat16], but got "
                            << input_type << "!";
  }
  auto target_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(target_type);
  auto target_elememt = target_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(target_elememt);
  auto target_type_id = target_elememt->type_id();
  if (input_type_id != target_type_id) {
    MS_EXCEPTION(TypeError)
      << "For primitive[BinaryCrossEntorpy], the input and target should have same type, but got: " << input_type
      << " and " << target_type << ".";
  }
  auto weight_type = input_args[kInputIndex2]->GetType();
  if (weight_type->type_id() != kMetaTypeNone) {
    auto weight_elememt = weight_type->cast<TensorTypePtr>()->element();
    MS_EXCEPTION_IF_NULL(weight_elememt);
    auto weight_type_id = weight_elememt->type_id();
    if (input_type_id != weight_type_id) {
      MS_EXCEPTION(TypeError) << "For primitive[BinaryCrossEntorpy], the input and weight should have same type, "
                                 "but got: "
                              << input_type << " and " << weight_type << ".";
    }
  }
  return input_type;
}

BaseShapePtr BinaryCrossEntropyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto target_shape_ptr = input_args[kInputIndex1]->GetShape();
  if (!input_shape_ptr->isa<abstract::NoShape>() && !target_shape_ptr->isa<abstract::NoShape>()) {
    auto &input_shape = input_shape_ptr->GetShapeVector();
    auto &target_shape = target_shape_ptr->GetShapeVector();
    bool is_dynamic = IsDynamic(input_shape) || IsDynamic(target_shape);
    if (!is_dynamic) {
      MS_CHECK_VALUE(input_shape == target_shape,
                     CheckAndConvertUtils::FormatCheckMsg("input_shape", input_shape, kEqual, target_shape, primitive));
    }
    auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
    if (!weight_shape_ptr->isa<abstract::NoShape>()) {
      auto &weight_shape = weight_shape_ptr->GetShapeVector();
      if (!IsBroadcastable(input_shape, weight_shape)) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                 << "', the shape of 'weight' can not broadcast to the shape of 'input'.";
      }
    }
  }
  auto reduction = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  if (reduction.has_value() && static_cast<Reduction>(reduction.value()) == Reduction::NONE) {
    return input_shape_ptr->Clone();
  }

  return std::make_shared<abstract::Shape>(ShapeVector{});
}

TypePtrList BinaryCrossEntropyFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_type = input_tensor->Dtype();
  auto input_type_id = input_type->type_id();
  if (valid_types_.find(input_type_id) == valid_types_.end()) {
    MS_EXCEPTION(TypeError) << "For Primitive[BinaryCrossEntorpy], the type of the input must be "
                               "[Float16, Float32, BFloat16], but got "
                            << input_type << "!";
  }
  const auto &target_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  auto target_type = target_tensor->Dtype();
  auto target_type_id = target_type->type_id();
  if (input_type_id != target_type_id) {
    MS_EXCEPTION(TypeError)
      << "For primitive[BinaryCrossEntorpy], the input and target should have same type, but got: " << input_type
      << " and " << target_type << ".";
  }
  if (input_values[kInputIndex2] != mindspore::kNone) {
    const auto &weight_tensor = input_values[kInputIndex2]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(weight_tensor);
    auto weight_type = weight_tensor->Dtype();
    auto weight_type_id = weight_type->type_id();
    if (input_type_id != weight_type_id) {
      MS_EXCEPTION(TypeError) << "For primitive[BinaryCrossEntorpy], the input and weight should have same type,"
                                 " but got: "
                              << input_type << " and " << weight_type << ".";
    }
  }
  return {input_type};
}

ShapeArray BinaryCrossEntropyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto &input_shape = input_tensor->shape();
  const auto &target_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(target_tensor);
  auto target_shape = target_tensor->shape();
  bool is_dynamic = IsDynamic(input_shape) || IsDynamic(target_shape);
  if (!is_dynamic) {
    MS_CHECK_VALUE(input_shape == target_shape,
                   CheckAndConvertUtils::FormatCheckMsg("input_shape", input_shape, kEqual, target_shape, primitive));
  }
  if (input_values[kInputIndex2] != mindspore::kNone) {
    const auto &weight_tensor = input_values[kIndex2]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(weight_tensor);
    auto weight_shape = weight_tensor->shape();
    if (!IsBroadcastable(input_shape, weight_shape)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the shape of 'weight' can not broadcast to the shape of 'input'.";
    }
  }
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
