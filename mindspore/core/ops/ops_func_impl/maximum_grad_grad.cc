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
#include "ops/ops_func_impl/maximum_grad_grad.h"

#include <memory>
#include <utility>

#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MaximumGradGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  const auto x_shape = input_args[kIndex0]->GetShape();
  const auto y_shape = input_args[kIndex1]->GetShape();
  const auto dx_shape = input_args[kIndex2]->GetShape();
  const auto dy_shape = input_args[kIndex3]->GetShape();
  const auto &x_shape_vec = x_shape->GetShapeVector();
  const auto &y_shape_vec = y_shape->GetShapeVector();
  const auto &dx_shape_vec = dx_shape->GetShapeVector();
  const auto &dy_shape_vec = dy_shape->GetShapeVector();

  const auto x_is_dynamic = IsDynamic(x_shape_vec);
  const auto y_is_dynamic = IsDynamic(y_shape_vec);
  const auto dx_is_dynamic = IsDynamic(dx_shape_vec);
  const auto dy_is_dynamic = IsDynamic(dy_shape_vec);

  constexpr size_t broadcast_num = 2;
  constexpr size_t output_num = 3;
  std::vector<abstract::BaseShapePtr> out_shape_list;
  std::vector<AbstractBasePtr> broadcast_args;
  out_shape_list.reserve(output_num);
  broadcast_args.reserve(broadcast_num);
  if (MS_UNLIKELY((x_is_dynamic && !dx_is_dynamic) || (IsDynamicRank(x_shape_vec) && !IsDynamicRank(dx_shape_vec)))) {
    out_shape_list.push_back(dx_shape->Clone());
    broadcast_args.push_back(input_args[kIndex2]);
  } else {
    out_shape_list.push_back(x_shape->Clone());
    broadcast_args.push_back(input_args[kIndex0]);
  }

  if (MS_UNLIKELY((y_is_dynamic && !dy_is_dynamic) || (IsDynamicRank(y_shape_vec) && !IsDynamicRank(dy_shape_vec)))) {
    out_shape_list.push_back(dy_shape->Clone());
    broadcast_args.push_back(input_args[kIndex3]);
  } else {
    out_shape_list.push_back(y_shape->Clone());
    broadcast_args.push_back(input_args[kIndex1]);
  }

  const auto sopd_grad_shape = BroadCastInferShape(primitive->name(), broadcast_args);
  out_shape_list.push_back(sopd_grad_shape);
  auto out_shape = std::make_shared<abstract::TupleShape>(std::move(out_shape_list));

  if (MS_UNLIKELY(x_is_dynamic || y_is_dynamic || dx_is_dynamic || dy_is_dynamic)) {
    return out_shape;
  }

  if (MS_UNLIKELY((x_shape_vec != dx_shape_vec) || (y_shape_vec != dy_shape_vec))) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', Its input 'dx', 'dy' should have same shape and equal to x and y shape, but got 'x' shape:" << x_shape_vec
      << " vs 'dx' shape: " << dx_shape_vec << ", 'y' shape:" << y_shape_vec << " vs 'dy' shape: " << dy_shape_vec;
  }
  return out_shape;
}

TypePtr MaximumGradGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto x_type = input_args[kIndex0]->GetType();
  const auto y_type = input_args[kIndex1]->GetType();
  std::vector<TypePtr> type_list{x_type->Clone(), y_type->Clone(), x_type->Clone()};
  return std::make_shared<Tuple>(std::move(type_list));
}
}  // namespace ops
}  // namespace mindspore
