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
#include "ops/ops_func_impl/group_norm_grad.h"
#include <memory>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GroupNormGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto x_shape = x_shape_ptr->GetShapeVector();
  const int64_t C = x_shape[kIndex1];
  ShapeVector out_shape{C};
  std::vector<BaseShapePtr> shapes_list;
  if (IsDynamicRank(x_shape)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  (void)shapes_list.emplace_back(x_shape_ptr->Clone());
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(out_shape));
  (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(out_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr GroupNormGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto dy_type = input_args[kInputIndex0]->GetType();
  auto x_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(dy_type);
  std::vector<TypePtr> types_list;
  types_list = {dy_type->Clone(), x_type->Clone(), x_type->Clone()};
  return std::make_shared<Tuple>(types_list);
}

}  // namespace ops
}  // namespace mindspore
