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
#include "ops/ops_func_impl/simple_infer.h"

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
  auto input_x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input_x_shape);

  auto x_shape = input_x_shape->GetShapeVector();
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  }

  auto x_rank = SizeToLong(x_shape.size());
  if (x_rank == 0) {
    x_rank = 1;
  }
  auto start_opt = GetScalarValue<int64_t>(input_args[1]->GetValue());
  auto end_opt = GetScalarValue<int64_t>(input_args[2]->GetValue());
  if (!start_opt.has_value() || !end_opt.has_value()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  }
  auto start_dim = start_opt.value();
  auto end_dim = end_opt.value();
  CheckAndConvertUtils::CheckInRange<int64_t>("start_dim", start_dim, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  CheckAndConvertUtils::CheckInRange<int64_t>("end_dim", end_dim, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  auto out_shape = FlattenShapeCalc(x_shape, start_dim, end_dim);
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr FlattenExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

TypePtrList FlattenExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();
  return {input_type};
}

ShapeArray FlattenExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);

  const auto x_shape = x_tensor->shape();
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_values.size()), kGreaterEqual, input_num,
                                           prim_name);

  auto x_rank = SizeToLong(x_shape.size());
  if (x_rank == 0) {
    x_rank = 1;
  }
  auto start_opt = GetScalarValue<int64_t>(input_values[kInputIndex1]);
  auto end_opt = GetScalarValue<int64_t>(input_values[kInputIndex2]);
  auto start_dim = start_opt.value();
  auto end_dim = end_opt.value();
  CheckAndConvertUtils::CheckInRange<int64_t>("start_dim", start_dim, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  CheckAndConvertUtils::CheckInRange<int64_t>("end_dim", end_dim, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  auto out_shape = FlattenShapeCalc(x_shape, start_dim, end_dim);
  return ShapeArray{
    out_shape,
  };
}
REGISTER_SIMPLE_INFER(kNameFlattenExt, FlattenExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
