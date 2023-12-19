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
#include <algorithm>
#include <memory>
#include <vector>

#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/ops_func_impl/diagonal.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeVector CalcuateDiagonalShape(size_t dim1, size_t dim2, size_t x_rank, const ShapeVector &input_shape) {
  ShapeVector out_shape;
  for (size_t tmp_dim = 0; tmp_dim < x_rank; tmp_dim++) {
    if (tmp_dim != dim1 && tmp_dim != dim2) {
      out_shape.push_back(input_shape[tmp_dim]);
    }
  }
  return out_shape;
}

BaseShapePtr DiagonalFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[0]->GetShape()->GetShapeVector();
  constexpr size_t kDimNum = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 4;
  const int64_t dyn_shape = abstract::Shape::kShapeDimAny;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto x_rank = input_shape.size();
  auto offset_opt = GetScalarValue<int64_t>(input_args[1]->GetValue());
  auto dim1_opt = GetScalarValue<int64_t>(input_args[2]->GetValue());
  auto dim2_opt = GetScalarValue<int64_t>(input_args[3]->GetValue());

  if (!dim1_opt.has_value() || !dim1_opt.has_value()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto dim1 = dim1_opt.value();
  auto dim2 = dim2_opt.value();
  CheckAndConvertUtils::CheckInRange<int64_t>("dim1", dim1, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  CheckAndConvertUtils::CheckInRange<int64_t>("dim2", dim2, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  if (x_rank < kDimNum) {
    MS_EXCEPTION(ValueError) << "For 'Diagonal', input must be at least 2-dimensional, but got : " << x_rank << ".";
  }
  auto tmp_dim1 = (dim1 < 0) ? dim1 + x_rank : dim1;
  auto tmp_dim2 = (dim2 < 0) ? dim2 + x_rank : dim2;
  if (tmp_dim1 == tmp_dim2) {
    MS_EXCEPTION(ValueError) << "For 'Diagonal', dim1 and dim2 cannot be identical, but got : dim1 =" << dim1
                             << " and dim2 = " << dim2 << ".";
  }
  auto out_shape = CalcuateDiagonalShape(tmp_dim1, tmp_dim2, x_rank, input_shape);
  int64_t dsize = dyn_shape;
  if (offset_opt.has_value()) {
    auto offset = offset_opt.value();
    if (input_shape[tmp_dim1] != dyn_shape && input_shape[tmp_dim2] != dyn_shape) {
      if (offset >= 0) {
        dsize = std::max<int64_t>(std::min(input_shape[tmp_dim1], input_shape[tmp_dim2] - offset), 0);
      } else {
        dsize = std::max<int64_t>(std::min(input_shape[tmp_dim1] + offset, input_shape[tmp_dim2]), 0);
      }
    }
  }
  out_shape.push_back(dsize);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DiagonalFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]->GetType());
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
