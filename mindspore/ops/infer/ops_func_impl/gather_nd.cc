/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/gather_nd.h"
#include <memory>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GatherNdFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  const auto &indices_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(indices_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  std::vector<int64_t> indices_new_shape = indices_shape;
  // make a scalar to tensor whose shape is (1,)
  if (indices_new_shape.size() == 0) {
    indices_new_shape.emplace_back(1);
  }

  auto input_x_rank = input_x_shape.size();
  auto indices_rank = indices_new_shape.size();
  auto indices_end_value = indices_new_shape[indices_rank - kIndex1];

  if (indices_end_value == abstract::Shape::kShapeDimAny) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  MS_CHECK_VALUE(
    SizeToLong(input_x_rank) >= indices_end_value,
    CheckAndConvertUtils::FormatCheckIntegerMsg("In GatherNd, the input of indices data", SizeToLong(input_x_rank),
                                                kGreaterEqual, indices_end_value, primitive));

  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < indices_rank - kIndex1; ++i) {
    output_shape.push_back(indices_new_shape[i]);
  }

  for (size_t i = LongToSize(indices_end_value); i < input_x_rank; ++i) {
    output_shape.push_back(input_x_shape[i]);
  }

  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr GatherNdFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
