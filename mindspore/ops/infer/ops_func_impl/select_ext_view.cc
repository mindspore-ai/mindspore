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

#include "infer/ops_func_impl/select_ext_view.h"

#include <vector>
#include <set>
#include <memory>
#include <utility>
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/op_name.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr SelectExtViewFuncImpl::InferShape(const PrimitivePtr &prim,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = prim->name();
  auto input_x_shape = input_args[0]->GetShape()->GetShapeVector();
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);

  if (IsDynamicRank(input_x_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto axis_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (!axis_value_opt.has_value()) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto axis_value = axis_value_opt.value();
  auto x_rank = SizeToLong(input_x_shape.size());

  MS_CHECK_VALUE(axis_value >= -x_rank && axis_value < x_rank, "dim value error. dim:" + std::to_string(axis_value) +
                                                                 ", dim should be in [" + std::to_string(-x_rank) +
                                                                 ", " + std::to_string(x_rank) + ").");
  axis_value = axis_value < 0 ? axis_value + x_rank : axis_value;
  auto output_shape = input_x_shape;
  output_shape.erase(output_shape.begin() + axis_value);
  return std::make_shared<abstract::TensorShape>(output_shape);
}

ShapeArray SelectExtViewFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  auto input_x_shape = x_tensor->shape();
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);

  auto axis_value_opt = GetScalarValue<int64_t>(input_values[kInputIndex1]);
  auto axis_value = axis_value_opt.value();
  auto x_rank = SizeToLong(input_x_shape.size());

  MS_CHECK_VALUE(axis_value >= -x_rank && axis_value < x_rank, "dim value error. dim:" + std::to_string(axis_value) +
                                                                 ", dim should be in [" + std::to_string(-x_rank) +
                                                                 ", " + std::to_string(x_rank) + ").");
  axis_value = axis_value < 0 ? axis_value + x_rank : axis_value;

  auto output_shape = input_x_shape;
  output_shape.erase(output_shape.begin() + axis_value);

  return {output_shape};
}

TypePtr SelectExtViewFuncImpl::InferType(const PrimitivePtr &prim,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  return x_type;
}

TypePtrList SelectExtViewFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}
}  // namespace ops
}  // namespace mindspore
