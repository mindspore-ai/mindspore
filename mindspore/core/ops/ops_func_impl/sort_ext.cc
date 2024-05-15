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

#include "ops/ops_func_impl/sort_ext.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

TypePtr SortExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto output0_type = input_args[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kInt8, kInt16, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", output0_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{output0_type, kInt64});
}

BaseShapePtr SortExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape());
  auto x_shape = shape_map[kShape];
  if (IsDynamicRank(x_shape)) {
    abstract::BaseShapePtr out_shape_ptr =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  }
  auto input_rank = SizeToLong(x_shape.size());

  int64_t dim_val = -1;
  auto dim = input_args[kInputIndex1]->GetValue();
  auto dim_opt = GetScalarValue<int64_t>(dim);
  if (dim_opt.has_value()) {
    dim_val = dim_opt.value();
  }
  auto min_dim = 0;
  auto max_dim = 0;
  if ((-input_rank) <= (input_rank - 1)) {
    min_dim = -input_rank;
    max_dim = input_rank - 1;
  } else {
    min_dim = input_rank - 1;
    max_dim = -input_rank;
  }
  CheckAndConvertUtils::CheckInRange<int64_t>("dim", dim_val, kIncludeBoth, {min_dim, max_dim}, prim_name);
  auto out_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
}

}  // namespace ops
}  // namespace mindspore
