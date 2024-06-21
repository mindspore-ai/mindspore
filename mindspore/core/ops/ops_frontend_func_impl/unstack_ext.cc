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

#include <utility>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore::ops {
namespace {
AbstractBasePtr InferOutputForRankKnown(const PrimitivePtr &primitive, const ShapeVector &input_shape,
                                        const ValuePtr &axis_value, const TypePtr &element_type) {
  size_t element_size = 1;
  size_t input_rank = input_shape.size();
  MS_CHECK_VALUE(input_rank > 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("input rank", input_rank, kGreaterThan, 0, primitive));
  auto axis_res = GetScalarValue<int64_t>(axis_value);
  bool dynamic_len = false;
  ShapeVector element_shape;
  if (MS_UNLIKELY(!axis_res.has_value())) {
    dynamic_len = true;
    element_shape = ShapeVector(input_rank - 1, abstract::TensorShape::kShapeDimAny);
  } else {
    auto axis_temp = axis_res.value();
    MS_CHECK_VALUE(-SizeToLong(input_rank) <= axis_temp && axis_temp < SizeToLong(input_rank),
                   CheckAndConvertUtils::FormatCheckInRangeMsg(
                     "axis", axis_temp, kIncludeLeft, {-SizeToLong(input_rank), SizeToLong(input_rank)}, primitive));
    auto axis = LongToSize(axis_temp < 0 ? SizeToLong(input_rank) + axis_temp : axis_temp);
    auto target_num = input_shape[axis];
    if (target_num == abstract::TensorShape::kShapeDimAny) {
      dynamic_len = true;
    } else {
      MS_CHECK_VALUE(target_num > 0,
                     CheckAndConvertUtils::FormatCheckIntegerMsg("out number", target_num, kGreaterThan, 0, primitive));
      element_size = LongToSize(target_num);
      // For Ascend only, move to backend pass latter.
      primitive->AddAttr("num", MakeValue(target_num));
    }
    element_shape.reserve(input_rank - 1);
    for (size_t i = 0; i < input_rank; ++i) {
      if (MS_UNLIKELY(i == axis)) {
        continue;
      }
      element_shape.push_back(input_shape[i]);
    }
  }

  abstract::AbstractBasePtrList out_tensors;
  out_tensors.reserve(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    out_tensors.push_back(std::make_shared<abstract::AbstractTensor>(element_type, element_shape));
  }

  auto out_tuple = std::make_shared<abstract::AbstractTuple>(out_tensors);
  if (MS_UNLIKELY(dynamic_len)) {
    MS_LOG(EXCEPTION) << "Unstack's output is dynamic tuple is not supported";
    out_tuple->CheckAndConvertToDynamicLenSequence();
  }
  return out_tuple;
}
}  // namespace
class UnstackExtBaseFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    auto input_type = input_args[kInputIndex0]->GetType();
    auto tensor_type = input_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element_type = tensor_type->element()->Clone();
    auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
      MS_LOG(EXCEPTION) << "Unstack's output is dynamic tuple is not supported";
      abstract::AbstractBasePtrList out_tensors = {
        std::make_shared<abstract::AbstractTensor>(element_type, ShapeVector{abstract::TensorShape::kShapeRankAny})};

      auto out_tuple = std::make_shared<abstract::AbstractTuple>(out_tensors);
      out_tuple->CheckAndConvertToDynamicLenSequence();
      return out_tuple;
    }

    auto axis_value = input_args[kInputIndex1]->GetValue();
    return InferOutputForRankKnown(primitive, input_shape, axis_value, element_type);
  }
};

class UnstackExtFrontendFuncImpl : public UnstackExtBaseFrontendFuncImpl {};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("UnstackExt", UnstackExtFrontendFuncImpl);
}  // namespace mindspore::ops
