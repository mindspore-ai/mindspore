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
#include "infer/ops_func_impl/unique_consecutive.h"
#include <functional>
#include <vector>
#include <memory>
#include <algorithm>
#include "utils/log_adapter.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore::ops {
ShapeArray UniqueConsecutiveFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto &x = input_infos[kInputIndex0];
  auto shape_x = x->GetShape();
  auto return_inverse = input_infos[kInputIndex1]->GetScalarValue<bool>();
  auto return_counts = input_infos[kInputIndex2]->GetScalarValue<bool>();
  auto empty_shape = ShapeVector{};
  if (x->IsDynamicRank()) {
    return {{abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}};
  }

  if (input_infos[kInputIndex3]->IsNone()) {
    auto y_max_shape =
      std::accumulate(shape_x.begin(), shape_x.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    auto out_shape = x->IsDynamic() ? ShapeVector{abstract::Shape::kShapeDimAny} : ShapeVector{y_max_shape};
    auto inverse_indices_shape = (return_inverse.has_value() && return_inverse.value()) ? shape_x : empty_shape;
    auto counts_shape = (return_counts.has_value() && return_counts.value()) ? out_shape : empty_shape;
    return {out_shape, inverse_indices_shape, counts_shape};
  }

  auto dim = input_infos[kInputIndex3]->GetScalarValue<int64_t>();
  if (!dim.has_value()) {
    auto itr_max_dim_shape = std::max_element(shape_x.begin(), shape_x.end());
    return {shape_x, ShapeVector{*itr_max_dim_shape}, ShapeVector{*itr_max_dim_shape}};
  }
  auto dim_value = dim.value();
  if (dim_value < -static_cast<int64_t>(shape_x.size()) || dim_value >= static_cast<int64_t>(shape_x.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value of 'dim' should be in ["
                             << -static_cast<int64_t>(shape_x.size()) << ", " << shape_x.size() << "), but got "
                             << dim_value;
  }
  if (dim_value < 0) {
    dim_value += static_cast<int64_t>(shape_x.size());
  }

  return {shape_x, ShapeVector{shape_x[dim_value]}, ShapeVector{shape_x[dim_value]}};
}

std::vector<TypeId> UniqueConsecutiveFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType(), kNumberTypeInt64, kNumberTypeInt64};
}

REGISTER_SIMPLE_INFER(kNameUniqueConsecutive, UniqueConsecutiveFuncImpl)
}  // namespace mindspore::ops
