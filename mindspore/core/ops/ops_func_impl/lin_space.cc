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

#include "ops/ops_func_impl/lin_space.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LinSpaceFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto start_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto start_shape = start_shape_ptr->GetShapeVector();
  auto stop_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto stop_shape = stop_shape_ptr->GetShapeVector();
  auto num_value_ptr = input_args[kInputIndex2]->GetValue();
  auto num_value = GetScalarValue<int64_t>(num_value_ptr);

  if (MS_UNLIKELY(IsDynamic(start_shape) || IsDynamic(stop_shape))) {
    ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::TensorShape>(infered_shape);
  }

  // 0-D tensor input.
  if (start_shape.empty() && stop_shape.empty()) {
    // Output is dynamic shape.
    if (!num_value.has_value()) {
      ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
      return std::make_shared<abstract::TensorShape>(infered_shape);
    } else {
      int64_t num = num_value.value();
      MS_CHECK_VALUE(num > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("num", num, kGreaterThan, 0, primitive));
      ShapeVector infered_shape{num};
      return std::make_shared<abstract::TensorShape>(infered_shape);
    }
  }

  // Support vmap.
  size_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = LongToSize(GetValue<int64_t>(value_ptr));
  }

  MS_CHECK_VALUE(
    start_shape.size() == batch_rank,
    CheckAndConvertUtils::FormatCheckIntegerMsg("rank of 'start'", start_shape.size(), kEqual, batch_rank, primitive));
  MS_CHECK_VALUE(
    stop_shape.size() == batch_rank,
    CheckAndConvertUtils::FormatCheckIntegerMsg("rank of 'stop'", stop_shape.size(), kEqual, batch_rank, primitive));
  MS_CHECK_VALUE(start_shape == stop_shape,
                 CheckAndConvertUtils::FormatCheckMsg("shape of 'start'", start_shape, kEqual, stop_shape, primitive));

  ShapeVector out_shape(start_shape.begin(), start_shape.end());
  if (!num_value.has_value()) {
    out_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    int64_t num = num_value.value();
    MS_CHECK_VALUE(num > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("num", num, kGreaterThan, 0, primitive));
    out_shape.push_back(num);
  }
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr LinSpaceFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto start_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(start_type);
  return start_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
