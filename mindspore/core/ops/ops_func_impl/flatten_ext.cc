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

#include "ops/ops_func_impl/flatten_ext.h"
#include <algorithm>
#include <functional>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/ir/value.h"
#include "mindapi/ir/primitive.h"

namespace mindspore {
namespace ops {
namespace {
ShapeVector FlattenShapeCalc(const ShapeVector &input_shape, const int64_t &start_dim, const int64_t &end_dim) {
  ShapeVector out_shape;
  int64_t dim_size = SizeToLong(input_shape.size());
  if (dim_size == 0) {
    return {1};
  }
  auto start_dim_fix = start_dim < 0 ? start_dim + dim_size : start_dim;
  auto end_dim_fix = end_dim < 0 ? end_dim + dim_size : end_dim;
  if (start_dim_fix > end_dim_fix) {
    MS_EXCEPTION(ValueError) << "For 'flatten', 'start_dim' cannot come after 'end_dim'.";
  }
  if (start_dim_fix == end_dim_fix) {
    return input_shape;
  }

  int64_t slice_numel = 1;
  for (int64_t i = start_dim_fix; i <= end_dim_fix; i++) {
    if (input_shape[i] == -1) {
      slice_numel = -1;
      break;
    }
    slice_numel = slice_numel * input_shape[i];
  }
  out_shape.reserve(dim_size - end_dim_fix + start_dim_fix);
  for (int64_t i = 0; i < start_dim_fix; i++) {
    out_shape.push_back(input_shape[i]);
  }
  out_shape.push_back(slice_numel);
  for (int64_t i = end_dim_fix + 1; i < dim_size; i++) {
    out_shape.push_back(input_shape[i]);
  }
  return out_shape;
}
}  // namespace

BaseShapePtr FlattenExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  if (input_x_shape->IsDimZero()) {
    MS_LOG(EXCEPTION) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  auto x_shape = input_x_shape->GetShapeVector();
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t num = 3;
  auto input_num = SizeToLong(input_args.size());
  MS_CHECK_VALUE(input_num == num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("input numbers", input_num, kEqual, num, primitive));
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  }

  auto x_rank = SizeToLong(x_shape.size());
  auto start_opt = GetScalarValue<int64_t>(input_args[1]->GetValue());
  auto end_opt = GetScalarValue<int64_t>(input_args[2]->GetValue());
  if (!start_opt.has_value() || !end_opt.has_value()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  }
  int64_t start_dim = start_opt.value();
  int64_t end_dim = end_opt.value();
  MS_CHECK_VALUE(
    -x_rank <= start_dim && start_dim < x_rank,
    CheckAndConvertUtils::FormatCheckInRangeMsg("start_dim", start_dim, kIncludeLeft, {-x_rank, x_rank}, primitive));
  MS_CHECK_VALUE(
    -x_rank <= end_dim && end_dim < x_rank,
    CheckAndConvertUtils::FormatCheckInRangeMsg("end_dim", end_dim, kIncludeLeft, {-x_rank, x_rank}, primitive));
  auto out_shape = FlattenShapeCalc(x_shape, start_dim, end_dim);
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr FlattenExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
