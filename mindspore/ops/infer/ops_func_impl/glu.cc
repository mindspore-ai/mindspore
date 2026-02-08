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

#include <vector>
#include "infer/ops_func_impl/glu.h"
#include "ops_utils/op_constants.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/op_name.h"

namespace mindspore {
namespace ops {
ShapeArray GLUFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kIndex0];
  if (input->IsDynamicRank()) {
    return {{abstract::Shape::kShapeRankAny}};
  }
  auto out_shape = input->GetShape();
  const auto dim_range = SizeToLong(out_shape.size());
  if (MS_UNLIKELY(dim_range == 0)) {
    MS_EXCEPTION(RuntimeError)
      << "For GLU, scalar input is not supported because halving size must be even, but got a scalar.";
  }
  const auto dim = input_infos[kIndex1]->GetScalarValue<int64_t>();
  if (!dim.has_value()) {
    return {ShapeVector(out_shape.size(), abstract::Shape::kShapeDimAny, ShapeVector::allocator_type())};
  }
  const auto raw_dim_val = dim.value();
  if (MS_UNLIKELY((raw_dim_val < -dim_range) || (raw_dim_val >= dim_range))) {
    MS_EXCEPTION(IndexError) << "For GLU, dimension should be in range of [" << -dim_range << ", " << (dim_range - 1)
                             << "], but got " << raw_dim_val << ".";
  }
  const auto dim_val = raw_dim_val < 0 ? raw_dim_val + dim_range : raw_dim_val;
  const auto sep_dim_size = out_shape[dim_val];
  if (sep_dim_size > 0) {
    if (MS_UNLIKELY(sep_dim_size % 2 != 0)) {
      MS_EXCEPTION(RuntimeError) << "For GLU, halving dimension must be even, but dimension " << raw_dim_val
                                 << " is size " << sep_dim_size << ".";
    }
    out_shape[dim_val] /= 2;
  }
  return {out_shape};
}

std::vector<TypeId> GLUFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
