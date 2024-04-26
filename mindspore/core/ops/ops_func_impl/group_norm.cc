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
#include "ops/ops_func_impl/group_norm.h"
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
constexpr int64_t kNumberTwo = 2;
constexpr int64_t kNumberEight = 8;
BaseShapePtr GroupNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto num_groups_opt = GetScalarValue<int64_t>(input_args[1]->GetValue());
  std::vector<BaseShapePtr> shapes_list;
  if (!num_groups_opt.has_value()) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  int64_t num_groups = num_groups_opt.value();
  if (IsDynamicRank(x_shape->GetShapeVector())) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  const auto x_rank = x_shape->GetShapeVector().size();
  if (x_rank < kNumberTwo || x_rank > kNumberEight) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', The dim of input must be between 2 and 8. But got: " << x_rank << ".";
  }
  const int64_t N = x_shape->GetShapeVector()[0];
  ShapeVector out_shape{N, num_groups};
  (void)shapes_list.emplace_back(x_shape->Clone());
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(out_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(out_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr GroupNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  std::vector<TypePtr> types_list;
  types_list = {x_type->Clone(), x_type->Clone(), x_type->Clone()};
  return std::make_shared<Tuple>(types_list);
}

}  // namespace ops
}  // namespace mindspore
