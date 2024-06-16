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

#include "ops/ops_func_impl/repeat_interleave_int.h"
#include <utility>
#include <memory>
#include <functional>
#include "ops/op_utils.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
inline ShapeVector GetInferredShape(const PrimitivePtr &primitive, const ShapeVector &input_shape,
                                    const int64_t &repeats, const ValuePtr &dim) {
  ShapeVector result_shape;
  auto rank = SizeToLong(input_shape.size());
  auto numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (!dim->isa<None>()) {
    auto dim_opt = GetScalarValue<int64_t>(dim);
    if (dim_opt.has_value()) {
      int64_t real_dim = dim_opt.value();
      MS_CHECK_VALUE(
        real_dim >= -rank && real_dim <= rank - 1,
        CheckAndConvertUtils::FormatCheckInRangeMsg("dim", real_dim, kIncludeBoth, {-rank, rank - 1}, primitive));
      real_dim = (real_dim < 0) ? (real_dim + rank) : real_dim;
      for (int64_t i = 0; i < rank; i++) {
        if (i == real_dim) {
          auto value = input_shape[i] == -1 ? -1 : repeats * input_shape[i];
          result_shape.emplace_back(value);
        } else {
          result_shape.emplace_back(input_shape[i]);
        }
      }
    } else {
      ShapeVector res_shape(rank, abstract::TensorShape::kShapeDimAny);
      return res_shape;
    }
  } else {
    result_shape.emplace_back(repeats * numel);
  }
  return result_shape;
}
}  // namespace

BaseShapePtr RepeatInterleaveIntFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  auto repeats_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());

  if (MS_UNLIKELY(IsDynamicRank(x_shape)) || MS_UNLIKELY(!repeats_opt.has_value())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto repeats = repeats_opt.value();
  auto dim = input_args[kInputIndex2]->GetValue();
  if (repeats < 0) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' can not be negative.";
  }

  auto inferred_shape = GetInferredShape(primitive, x_shape, repeats, dim);
  return std::make_shared<abstract::TensorShape>(inferred_shape);
}

TypePtr RepeatInterleaveIntFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

ShapeArray RepeatInterleaveIntFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto x_shape = x_tensor->shape();
  auto repeats_opt = GetScalarValue<int64_t>(input_values[kInputIndex1]);

  auto repeats = repeats_opt.value();
  auto dim = input_values[kInputIndex2];
  if (repeats < 0) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', 'repeats' can not be negative.";
  }

  auto inferred_shape = GetInferredShape(primitive, x_shape, repeats, dim);
  return ShapeArray{
    inferred_shape,
  };
}

TypePtrList RepeatInterleaveIntFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_x_type = x_tensor->Dtype();
  TypePtrList type_ptr_list{input_x_type};
  return type_ptr_list;
}
REGISTER_SIMPLE_INFER(kNameRepeatInterleaveInt, RepeatInterleaveIntFuncImpl)
}  // namespace ops
}  // namespace mindspore
