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

#include <map>
#include <string>
#include "ops/ops_func_impl/lin_space_ext.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LinSpaceExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto steps_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (!(CheckAndConvertUtils::IsTensor(input_args[kInputIndex0]) &&
        CheckAndConvertUtils::IsTensor(input_args[kInputIndex1]))) {
    if (!MS_LIKELY(steps_opt.has_value())) {
      ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
      return std::make_shared<abstract::TensorShape>(infered_shape);
    } else {
      int64_t steps = steps_opt.value();
      MS_CHECK_VALUE(steps > 0,
                     CheckAndConvertUtils::FormatCheckIntegerMsg("steps", steps, kGreaterThan, 0, primitive));
      ShapeVector infered_shape{steps};
      return std::make_shared<abstract::TensorShape>(infered_shape);
    }
  }

  const auto &start_shape_ptr = input_args[kInputIndex0]->GetShape();
  const auto &start_shape = start_shape_ptr->GetShapeVector();
  const auto &end_shape_ptr = input_args[kInputIndex1]->GetShape();
  const auto &end_shape = end_shape_ptr->GetShapeVector();
  const auto &steps_value_ptr = input_args[kInputIndex2]->GetValue();
  const auto &steps_value = GetScalarValue<int64_t>(steps_value_ptr);
  if (MS_UNLIKELY(IsDynamic(start_shape) || IsDynamic(end_shape))) {
    ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::TensorShape>(infered_shape);
  }
  // 0-D tensor input.
  if (start_shape.empty() && end_shape.empty()) {
    // Output is dynamic shape.
    if (!steps_value.has_value()) {
      ShapeVector infered_shape{abstract::Shape::kShapeDimAny};
      return std::make_shared<abstract::TensorShape>(infered_shape);
    } else {
      int64_t steps = steps_value.value();
      MS_CHECK_VALUE(steps > 0,
                     CheckAndConvertUtils::FormatCheckIntegerMsg("steps", steps, kGreaterThan, 0, primitive));
      ShapeVector infered_shape{steps};
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
  MS_CHECK_VALUE(end_shape.size() == batch_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                   "rank of 'end'", end_shape.size(), kEqual, batch_rank, primitive));
  MS_CHECK_VALUE(start_shape == end_shape,
                 CheckAndConvertUtils::FormatCheckMsg("shape of 'start'", start_shape, kEqual, end_shape, primitive));

  ShapeVector out_shape(start_shape.begin(), start_shape.end());
  if (!steps_value.has_value()) {
    out_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    int64_t steps = steps_value.value();
    MS_CHECK_VALUE(steps > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("steps", steps, kGreaterThan, 0, primitive));
    out_shape.push_back(steps);
  }
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr LinSpaceExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);

  auto start_dtype = input_args[kInputIndex0]->GetType();
  auto end_dtype = input_args[kInputIndex1]->GetType();
  if (CheckAndConvertUtils::IsTensor(input_args[kInputIndex0]) ||
      CheckAndConvertUtils::IsTensor(input_args[kInputIndex1])) {
    std::map<std::string, TypePtr> type_dict = {
      {"start type", start_dtype},
      {"end type", end_dtype},
    };
    (void)CheckAndConvertUtils::CheckTensorTypeSame(type_dict, common_valid_types_with_bool, primitive->name());
  }
  TypeId type_id;
  if (input_args[kInputIndex3]->GetType()->isa<TypeNone>()) {
    type_id = kFloat32->type_id();
  } else {
    auto dtype_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: dtype input should have valid value.");
    type_id = static_cast<TypeId>(dtype_opt.value());
  }
  return std::make_shared<TensorType>(TypeIdToType(type_id));
}
}  // namespace ops
}  // namespace mindspore
