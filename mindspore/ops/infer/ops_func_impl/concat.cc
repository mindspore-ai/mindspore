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

#include "infer/ops_func_impl/concat.h"
#include <map>
#include <algorithm>

#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <optional>
#include "primitive/op_name.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore::ops {
namespace {
constexpr int64_t kUnknownDiffIdx = std::numeric_limits<int64_t>().max();

inline size_t NormalizeAxis(int64_t axis, size_t rank) {
  return LongToSize(axis >= 0 ? axis : (axis + SizeToLong(rank)));
}

inline std::pair<ShapeVector, int64_t> CheckShapesValid(const ShapeArray &shapes, const PrimitivePtr &primitive) {
  // 1. Only one axis different: kShapeDimAny is considered as same.
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
    MS_CHECK_VALUE(output_shape.size() == shape.size(),
                   CheckAndConvertUtils::FormatCommMsg("For primitive[", primitive->name(),
                                                       "], element rank must be same(shapes are ", shapes, ")!"));
    for (size_t j = 0; j < output_shape.size(); ++j) {
      if (output_shape[j] == abstract::TensorShape::kShapeDimAny) {
        output_shape[j] = shape[j];
      } else if (shape[j] != abstract::TensorShape::kShapeDimAny && output_shape[j] != shape[j]) {
        auto new_diff_idx = SizeToLong(j);
        if (diff_idx == kUnknownDiffIdx) {
          diff_idx = new_diff_idx;
        } else if (diff_idx != new_diff_idx) {
          MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                                   << "] only support one axis different, bug got more than one(shapes is " << shapes
                                   << ")!";
        }
      }
    }
  }

  return std::make_pair(output_shape, diff_idx);
}

inline ShapeVector CalOutputShapeInDynamicLenCase(ShapeVector key_shape, std::optional<int64_t> axis_res,
                                                  const PrimitivePtr &primitive) {
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
                               << "], shapes of tensors must match except in given concat axis " << axis
                               << ", but got mismatch in " << LongToSize(diff_idx) << "(all input shapes are " << shapes
                               << ")!";
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

ShapeArray ConcatFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto axis = input_infos.back()->GetScalarValue<int64_t>();
  auto &tensors = input_infos[0];
  ShapeVector output_shape;
  if (MS_LIKELY(tensors->IsSequence())) {
    if (MS_UNLIKELY(tensors->IsDynamicSequence())) {
      output_shape = CalOutputShapeInDynamicLenCase(tensors->GetDynamicSequenceElement()->GetShape(), axis, primitive);
    } else {
      auto elements = tensors->GetSequenceElements();
      ShapeArray shapes;
      std::transform(elements.begin(), elements.end(), std::back_inserter(shapes),
                     [](const InferInfoPtr &info) { return info->GetShape(); });
      output_shape = CheckAndCalOutputShapeInTupleCase(shapes, axis, primitive);
    }
  } else {
    ShapeArray shapes;
    std::transform(input_infos.begin(), input_infos.end() - 1, std::back_inserter(shapes),
                   [](const InferInfoPtr &info) { return info->GetShape(); });
    output_shape = CheckAndCalOutputShapeInTupleCase(shapes, axis, primitive);
  }
  return {output_shape};
}

std::vector<TypeId> ConcatFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  std::vector<TypePtr> element_types;
  auto &input = input_infos[kInputIndex0];
  if (input->IsSequence()) {
    if (MS_UNLIKELY(input->IsDynamicSequence())) {
      auto element_type = input->GetDynamicSequenceElement()->GetType();
      CheckAndConvertUtils::CheckTypeIdValid("tensors", element_type, common_valid_type_ids_with_complex_and_bool,
                                             primitive->name());
      return {element_type};
    } else {
      const auto &elements = input->GetSequenceElements();
      (void)std::transform(elements.begin(), elements.end(), std::back_inserter(element_types),
                           [](const auto &info) { return TypeIdToType(info->GetType()); });
    }
  } else {
    (void)std::transform(input_infos.begin(), input_infos.end() - 1, std::back_inserter(element_types),
                         [](const auto &info) { return TypeIdToType(info->GetType()); });
  }
  MS_CHECK_VALUE(element_types.size() > 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                             "elements size", element_types.size(), kGreaterThan, 0, primitive));
  auto out_type = element_types[0];
  for (TypePtr element_type : element_types) {
    out_type = PromoteType(out_type, element_type, primitive->name());
  }
  return {out_type->type_id()};
}

}  // namespace mindspore::ops
