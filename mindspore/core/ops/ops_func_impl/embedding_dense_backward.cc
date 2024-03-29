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

#include "ops/ops_func_impl/embedding_dense_backward.h"
#include <vector>
#include <memory>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingDenseBackwardFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto grad_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto num_weights = input_args[kIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(num_weights);
  auto num_weights_opt = GetScalarValue<int64_t>(num_weights);

  auto first_dim = num_weights_opt.has_value() ? num_weights_opt.value() : abstract::Shape::kShapeDimAny;
  auto second_dim = IsDynamicRank(grad_shape) ? abstract::Shape::kShapeDimAny : grad_shape.back();

  return std::make_shared<abstract::Shape>(ShapeVector{first_dim, second_dim});
}

TypePtr EmbeddingDenseBackwardFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
