/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/quantile_ext.h"

#include <algorithm>
#include <vector>
#include "utils/shape_utils.h"

namespace mindspore::ops {
ShapeArray QuantileExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_shape = input_infos[0]->GetShape();
  const auto q_shape = input_infos[1]->GetShape();
  if (IsDynamicRank(q_shape) || IsDynamicRank(input_shape)) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  if (q_shape.size() > 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', q must be a scalar or one-dimensional Tensor.";
  }
  const auto keepdim = input_infos[3]->GetScalarValue<bool>();
  if (!keepdim.has_value()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  ShapeVector output_shape;
  if (input_infos[2]->IsNone()) {
    if (keepdim.value()) {
      output_shape.assign(input_shape.size(), 1);
    }
  } else {
    const auto dim = input_infos[2]->GetScalarValue<int64_t>();
    if (!dim.has_value()) {
      return {{abstract::TensorShape::kShapeRankAny}};
    }
    const auto rank = std::max<int64_t>(static_cast<int64_t>(input_shape.size()), 1);
    if (dim.value() < -rank || dim.value() >= rank) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', dim must be in [" << -rank << ", " << rank
                               << ").";
    }
    output_shape = input_shape;
    if (!input_shape.empty()) {
      const auto axis = dim.value() < 0 ? dim.value() + rank : dim.value();
      if (keepdim.value()) {
        output_shape[axis] = 1;
      } else {
        output_shape.erase(output_shape.begin() + axis);
      }
    }
  }
  if (!q_shape.empty()) {
    output_shape.insert(output_shape.begin(), q_shape[0]);
  }
  return {output_shape};
}

std::vector<TypeId> QuantileExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  const auto dtype = input_infos[0]->GetType();
  if (dtype != kNumberTypeFloat32 && dtype != kNumberTypeFloat64) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input must have dtype float32 or float64.";
  }
  if (input_infos[1]->GetType() != dtype) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input and q must have the same dtype.";
  }
  return {dtype};
}
}  // namespace mindspore::ops
