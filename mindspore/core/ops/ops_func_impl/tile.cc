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

#include "ops/ops_func_impl/tile.h"

#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/dtype/number.h"

namespace mindspore::ops {
namespace {
ShapeVector ExpandInHeadIfNeed(const ShapeVector &tag_shape, size_t expect_len) {
  auto tag_rank = tag_shape.size();
  if (expect_len < tag_rank) {
    MS_LOG(EXCEPTION) << "For 'Tile', the rank of input: " << expect_len
                      << " should be greater than or equal to multiple's length: " << tag_rank << ".";
  }
  if (expect_len == tag_rank) {
    return tag_shape;
  }

  int64_t expand_value = IsDynamicRank(tag_shape) ? -1 : 1;
  ShapeVector res_shape(expect_len, expand_value);
  auto offset = expect_len - tag_rank;
  for (size_t i = 0; i < tag_rank; ++i) {
    res_shape[i + offset] = tag_shape[i];
  }

  return res_shape;
}
}  // namespace
BaseShapePtr TileFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto multiples_base_shape = input_args[kInputIndex1]->GetShape();
  // Multiples is Tuple[int] with dynamic length.
  if (MS_UNLIKELY(multiples_base_shape->isa<abstract::DynamicSequenceShape>())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto multiples_value = input_args[kInputIndex1]->GetValue();
  // Multiples is Tensor, but unknown value.
  if (MS_UNLIKELY(multiples_base_shape->isa<abstract::TensorShape>() && !IsValueKnown(multiples_value))) {
    const auto &multiples_shape = multiples_base_shape->GetShapeVector();
    MS_CHECK_VALUE(multiples_shape.size() == 1,
                   CheckAndConvertUtils::FormatCheckIntegerMsg<int64_t>("the size of multiples", multiples_shape.size(),
                                                                        kEqual, 1, primitive));
    if (IsDynamic(multiples_shape)) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
    }
    return std::make_shared<abstract::TensorShape>(
      ShapeVector(multiples_shape[0], abstract::TensorShape::kShapeDimAny));
  }

  // A Tensor or a Tuple should be a value known one now.
  auto multiples_array_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  MS_CHECK_VALUE(multiples_array_opt.has_value(),
                 CheckAndConvertUtils::FormatCommMsg("For primitive[Tile], the multiples must has value here."));
  auto multiples_array = multiples_array_opt.value();
  auto multiples_len = multiples_array.size();

  ShapeVector inferred_shape;
  inferred_shape.reserve(multiples_len);

  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  const auto &x_shape = x_base_shape->GetShapeVector();
  auto adapted_shape = ExpandInHeadIfNeed(x_shape, multiples_len);
  for (size_t i = 0; i < multiples_len; ++i) {
    if (adapted_shape[i] < 0 || multiples_array.IsValueUnknown(i)) {
      inferred_shape.push_back(abstract::Shape::kShapeDimAny);
      continue;
    }

    if (multiples_array[i] < 0) {
      MS_EXCEPTION(ValueError) << "For 'Tile', 'multiples' must be an positive integer, but got " << multiples_array[i]
                               << ".";
    }

    inferred_shape.push_back(multiples_array[i] * adapted_shape[i]);
  }
  return std::make_shared<abstract::Shape>(inferred_shape);
}

TypePtr TileFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace mindspore::ops
