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

#include "ops/ops_func_impl/concat.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <optional>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore::ops {
namespace {
constexpr int64_t kUnknownDiffIdx = std::numeric_limits<int64_t>().max();

inline size_t NormalizeAxis(int64_t axis, size_t rank) {
  return LongToSize(axis >= 0 ? axis : (axis + SizeToLong(rank)));
}

inline std::pair<ShapeVector, int64_t> CheckShapesValid(const ShapeArray &shapes, const PrimitivePtr &primitive) {
  // 1. Only one dim different: kShapeDimAny is considered as same.
  // 2. all elements' rank should be same.

  int64_t diff_idx = kUnknownDiffIdx;
  ShapeVector output_shape = {abstract::TensorShape::kShapeRankAny};
  bool seen_rank_valid = false;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto &shape = shapes[i];
    if (IsDynamicRank(shape)) {
      continue;
    }

    if (!seen_rank_valid) {
      seen_rank_valid = true;
      output_shape = shape;
      continue;
    }

    MS_CHECK_VALUE(shape.size() > 0,
                   CheckAndConvertUtils::FormatCommMsg(
                     "For primitive[", primitive->name(),
                     "], all elements should not be zero rank, but got zero rank in position ", i, "!"));
    MS_CHECK_VALUE(
      output_shape.size() == shape.size(),
      CheckAndConvertUtils::FormatCommMsg("For primitive[", primitive->name(), "], element size must be same(",
                                          output_shape, " vs ", shape, ")!"));
    for (size_t j = 0; j < output_shape.size(); ++j) {
      if (output_shape[j] == abstract::TensorShape::kShapeDimAny) {
        output_shape[j] = shape[j];
      } else if (shape[j] != abstract::TensorShape::kShapeDimAny && output_shape[j] != shape[j]) {
        auto new_diff_idx = SizeToLong(j);
        if (diff_idx == kUnknownDiffIdx) {
          diff_idx = new_diff_idx;
        } else if (diff_idx != new_diff_idx) {
          MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                                   << "] only support one dim different, bug got more than one(shapes is " << shapes
                                   << ")!";
        }
      }
    }
  }

  return std::make_pair(output_shape, diff_idx);
}

inline ShapeVector CalOutputShapeInDynamicLenCase(const BaseShapePtr &base_shape, std::optional<int64_t> axis_res,
                                                  const PrimitivePtr &primitive) {
  auto dynamic_sequence = base_shape->cast<abstract::DynamicSequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(dynamic_sequence);
  auto element_base_shape = dynamic_sequence->element_shape();
  MS_EXCEPTION_IF_NULL(element_base_shape);
  auto key_shape = element_base_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(key_shape))) {
    key_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
  } else if (MS_UNLIKELY(!axis_res.has_value())) {
    key_shape = ShapeVector(key_shape.size(), abstract::TensorShape::kShapeDimAny);
  } else {
    auto axis_temp = axis_res.value();
    auto x_rank = SizeToLong(key_shape.size());
    MS_CHECK_VALUE(
      -x_rank <= axis_temp && axis_temp < x_rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("axis value", axis_temp, kIncludeLeft, {-x_rank, x_rank}, primitive));
    auto axis = NormalizeAxis(axis_temp, key_shape.size());
    key_shape[axis] = abstract::TensorShape::kShapeDimAny;
  }
  return key_shape;
}

inline ShapeVector CheckAndCalOutputShapeInTupleCase(const ShapeArray &shapes, std::optional<int64_t> axis_res,
                                                     const PrimitivePtr &primitive) {
  auto [output_shape, diff_idx] = CheckShapesValid(shapes, primitive);
  if (MS_UNLIKELY(IsDynamicRank(output_shape))) {
    return output_shape;
  }

  size_t axis;
  if (MS_UNLIKELY(!axis_res.has_value())) {
    if (diff_idx == kUnknownDiffIdx) {
      return ShapeVector(output_shape.size(), abstract::TensorShape::kShapeDimAny);
    }
    axis = LongToSize(diff_idx);
  } else {
    auto axis_temp = axis_res.value();
    auto x_rank = SizeToLong(output_shape.size());
    MS_CHECK_VALUE(
      -x_rank <= axis_temp && axis_temp < x_rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("axis value", axis_temp, kIncludeLeft, {-x_rank, x_rank}, primitive));
    axis = NormalizeAxis(axis_temp, output_shape.size());
    if (MS_UNLIKELY(diff_idx != kUnknownDiffIdx && axis != LongToSize(diff_idx))) {
      MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                               << "], the only different dim should be same with the axis, but got "
                               << LongToSize(diff_idx) << " and " << axis << "!";
    }
  }

  output_shape[axis] = 0;
  for (const auto &shape : shapes) {
    if (MS_UNLIKELY(IsDynamicRank(shape) || shape[axis] == abstract::TensorShape::kShapeDimAny)) {
      output_shape[axis] = abstract::TensorShape::kShapeDimAny;
      break;
    }
    output_shape[axis] += shape[axis];
  }

  return output_shape;
}
}  // namespace
BaseShapePtr ConcatFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto axis_index = input_args.size() - 1;
  auto axis_value = input_args[axis_index]->GetValue();
  auto axis_res = GetScalarValue<int64_t>(axis_value);

  ShapeVector output_shape;
  if (MS_LIKELY(!CheckAndConvertUtils::IsTensor(input_args[kInputIndex0]))) {
    auto x_base_shape = input_args[kInputIndex0]->GetShape();
    if (MS_UNLIKELY(x_base_shape->isa<abstract::DynamicSequenceShape>())) {
      output_shape = CalOutputShapeInDynamicLenCase(x_base_shape, axis_res, primitive);
    } else {
      auto tuple_shape = x_base_shape->cast<abstract::TupleShapePtr>();
      MS_EXCEPTION_IF_NULL(tuple_shape);
      ShapeArray shapes;
      shapes.reserve(tuple_shape->size());
      for (size_t i = 0; i < tuple_shape->size(); ++i) {
        shapes.push_back((*tuple_shape)[i]->GetShapeVector());
      }
      output_shape = CheckAndCalOutputShapeInTupleCase(shapes, axis_res, primitive);
    }
  } else {
    ShapeArray shapes;
    for (size_t i = 0; i < axis_index; ++i) {
      shapes.push_back(input_args[i]->GetShape()->GetShapeVector());
    }
    output_shape = CheckAndCalOutputShapeInTupleCase(shapes, axis_res, primitive);
  }

  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr ConcatFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto tuple_type = input_args[kInputIndex0]->GetType()->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_type);
  if (MS_UNLIKELY(tuple_type->dynamic_len())) {
    auto element_type = tuple_type->dynamic_element_type();
    MS_EXCEPTION_IF_NULL(element_type);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("dynamic-length element", element_type,
                                                     common_valid_types_with_complex_and_bool, primitive->name());
    return element_type->Clone();
  }

  auto elements = tuple_type->elements();
  MS_CHECK_VALUE(elements.size() > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("elements size", elements.size(),
                                                                                  kGreaterThan, 0, primitive));
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < elements.size(); ++i) {
    std::string element = "element" + std::to_string(i);
    (void)types.emplace(element, elements[i]);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, primitive->name());
  return elements[0]->Clone();
}
}  // namespace mindspore::ops
