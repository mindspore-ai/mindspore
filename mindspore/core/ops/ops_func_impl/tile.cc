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

#include <algorithm>
#include <memory>
#include "ir/functor.h"
#include "mindapi/base/shape_vector.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
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
ShapeVector ToMultiplesVector(const ArrayValue<int64_t> &array_value) {
  auto len = array_value.size();
  ShapeVector multiples_vec;
  multiples_vec.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    if (array_value.IsValueUnknown(i)) {
      multiples_vec.push_back(abstract::Shape::kShapeDimAny);
      continue;
    }

    if (array_value[i] < 0) {
      MS_EXCEPTION(ValueError) << "For 'Tile', 'dims' cannot contain negative integer numbers, but got "
                               << array_value[i] << " in " << i << "th.";
    }
    multiples_vec.push_back(array_value[i]);
  }

  return multiples_vec;
}
}  // namespace
void AdaptShapeAndMultipies(ShapeVector *shape, ShapeVector *dims) {
  MS_EXCEPTION_IF_NULL(shape);
  if (MS_UNLIKELY(IsDynamicRank(*shape))) {
    MS_LOG(INTERNAL_EXCEPTION) << "Shape should not be dynamic rank!";
  }
  MS_EXCEPTION_IF_NULL(dims);

  auto rank = shape->size();
  auto len = dims->size();
  if (len == rank) {
    return;
  }

  auto expect_len = std::max(rank, len);
  auto ExpandInHeadIfNeed = [](ShapeVector *vec, size_t length) -> void {
    if (vec->size() == length) {
      return;
    }

    auto offset = length - vec->size();
    ShapeVector res;
    vec->reserve(length);
    vec->insert(vec->begin(), offset, 1);
  };

  ExpandInHeadIfNeed(shape, expect_len);
  ExpandInHeadIfNeed(dims, expect_len);
}

BaseShapePtr TileFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  // The output rank is determined by data's rank and dims's length.
  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto dims_base_shape = input_args[kInputIndex1]->GetShape();
  if (MS_UNLIKELY(dims_base_shape->isa<abstract::DynamicSequenceShape>())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto dims_array_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  MS_CHECK_VALUE(dims_array_opt.has_value(),
                 CheckAndConvertUtils::FormatCommMsg("For primitive[Tile], the dims must has value here."));
  auto dims_array = dims_array_opt.value();
  auto dims = ToMultiplesVector(dims_array);

  AdaptShapeAndMultipies(&x_shape, &dims);
  auto adapted_rank = x_shape.size();
  ShapeVector inferred_shape;
  inferred_shape.reserve(adapted_rank);
  for (size_t i = 0; i < adapted_rank; ++i) {
    if (x_shape[i] == abstract::Shape::kShapeDimAny || dims[i] == abstract::Shape::kShapeDimAny) {
      inferred_shape.push_back(abstract::Shape::kShapeDimAny);
      continue;
    }

    inferred_shape.push_back(dims[i] * x_shape[i]);
  }
  return std::make_shared<abstract::TensorShape>(inferred_shape);
}

TypePtr TileFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace mindspore::ops
