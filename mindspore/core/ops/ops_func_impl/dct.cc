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
#include <utility>
#include "ops/ops_func_impl/dct.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr DCTFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = x_shape->GetShapeVector();
  if (IsDynamicRank(x_shape_vec)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto rank = SizeToLong(x_shape_vec.size());

  // get value of n
  auto n = input_args[kIndex2]->GetValue();
  auto n_opt = GetScalarValue<int64_t>(n);

  // get value of axis
  auto axis = input_args[kIndex3]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  int64_t positive_axis = -1;
  if (axis_opt.has_value()) {
    auto axis_value = axis_opt.value();
    positive_axis = axis_value < 0 ? axis_value + rank : axis_value;
  } else {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  // infer shape with known axis
  ShapeVector out_shape;
  out_shape.assign(x_shape_vec.begin(), x_shape_vec.end());
  if (n_opt.has_value()) {
    auto n_value = n_opt.value();
    if (n_value == -1) {
      return x_shape->Clone();
    }
    out_shape[positive_axis] = n_value;
  } else {
    out_shape[positive_axis] = abstract::Shape::kShapeDimAny;
  }

  return std::make_shared<abstract::TensorShape>(std::move(out_shape));
}

TypePtr DCTFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  auto input_type_id = input_type->cast<TensorTypePtr>()->element()->type_id();

  static const std::vector<TypeId> double_type = {kNumberTypeFloat64, kNumberTypeComplex128, kNumberTypeUInt64,
                                                  kNumberTypeInt64};
  bool is_double_type = std::any_of(double_type.begin(), double_type.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_double_type) {
    return std::make_shared<TensorType>(kFloat64);
  } else {
    return std::make_shared<TensorType>(kFloat32);
  }
}

int32_t DCTFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto check_status = OP_CHECK_SUCCESS;
  auto x_shape_vec = input_args[kIndex0]->GetShape()->GetShapeVector();
  int64_t x_rank = SizeToLong(x_shape_vec.size());
  (void)CheckAndConvertUtils::FormatCheckInRangeMsg("rank of 'x'", x_rank, kIncludeBoth, {1, 8}, primitive);

  if (x_rank == 1 && x_shape_vec[0] == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  auto n = input_args[kIndex2]->GetValue();
  auto n_opt = GetScalarValue<int64_t>(n);
  if (n_opt.has_value()) {
    auto n_value = n_opt.value();
    if (n_value != -1) {
      (void)CheckAndConvertUtils::CheckInteger("n", n_value, kGreaterThan, 0, primitive->name());
    }
  }

  // These situations need to be handled in the kernel: x is dynamic or axes is None
  auto axis = input_args[kIndex3]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec) || !axis_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    auto axis_value = axis_opt.value();
    MS_CHECK_VALUE(
      axis_value >= -x_rank && axis_value < x_rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_value, kIncludeLeft, {-x_rank, x_rank}, primitive));
  }
  return check_status;
}

}  // namespace ops
}  // namespace mindspore
