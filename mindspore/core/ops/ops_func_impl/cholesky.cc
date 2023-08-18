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
#include "ops/ops_func_impl/cholesky.h"

namespace mindspore {
namespace ops {
BaseShapePtr CholeskyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto x_shape = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto const &x_shape_list = x_shape->GetShapeVector();
  // support dynamic rank
  if (IsDynamicRank(x_shape_list)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  const size_t x_dim = x_shape_list.size();
  constexpr size_t kDefaultRank = 2;
  constexpr size_t kRowIndex = 2;
  constexpr size_t kColIndex = 1;
  if (x_dim < kDefaultRank) {
    MS_EXCEPTION(ValueError) << "For Cholesky, the dimension of input x must be greater than or "
                             << "equal to 2"
                             << ", but got a " << x_dim << "-D Tensor.";
  }
  // support dynamic shape
  if (x_shape_list[x_dim - kColIndex] == abstract::Shape::kShapeDimAny ||
      x_shape_list[x_dim - kRowIndex] == abstract::Shape::kShapeDimAny) {
    auto new_shape_list = x_shape_list;
    if (x_shape_list[x_dim - kColIndex] != abstract::Shape::kShapeDimAny) {
      new_shape_list[x_dim - kRowIndex] = new_shape_list[x_dim - kColIndex];
      return std::make_shared<abstract::Shape>(new_shape_list);
    }
    if (x_shape_list[x_dim - kRowIndex] != abstract::Shape::kShapeDimAny) {
      new_shape_list[x_dim - kColIndex] = new_shape_list[x_dim - kRowIndex];
      return std::make_shared<abstract::Shape>(new_shape_list);
    }
    new_shape_list[x_dim - kColIndex] = abstract::Shape::kShapeDimAny;
    new_shape_list[x_dim - kRowIndex] = abstract::Shape::kShapeDimAny;
    return std::make_shared<abstract::Shape>(new_shape_list);
  }
  if (IsDynamic(x_shape_list)) {
    return x_shape->Clone();
  }
  if (x_shape_list[x_dim - kColIndex] != x_shape_list[x_dim - kRowIndex]) {
    MS_EXCEPTION(ValueError) << "For Cholesky, input x must be batch squares"
                             << ", but got batch " << x_shape_list[x_dim - kRowIndex] << " x "
                             << x_shape_list[x_dim - kColIndex] << " matrices.";
  }
  return x_shape->Clone();
}

TypePtr CholeskyFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  auto x_type = input_args[kIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
