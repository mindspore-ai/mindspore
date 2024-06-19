/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/grid_sampler_3d.h"

#include <algorithm>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GridSampler3DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto input_x_base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input_x_base_shape);
  auto input_x_shape = input_x_base_shape->GetShapeVector();

  auto grid_base_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(grid_base_shape);
  auto grid_shape = grid_base_shape->GetShapeVector();

  // dynamic rank
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{
      abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
      abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny});
  }
  // dynamic shape
  if (IsDynamicRank(input_x_shape)) {
    input_x_shape = {grid_shape[0], abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                     abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
  }
  if (IsDynamicRank(grid_shape)) {
    grid_shape = {input_x_shape[0], abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                  abstract::TensorShape::kShapeDimAny, 3};
  }
  const size_t kFive = 5;
  if (input_x_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'input_x' must be a 5-D tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-D tensor.";
  }
  if (grid_shape.size() != kFive) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'grid' must be a 5-D tensor, but got "
                             << std::to_string(grid_shape.size()) << "-D tensor.";
  }
  if (!IsDynamic(input_x_shape) && !IsDynamic(grid_shape)) {
    if (input_x_shape[kInputIndex0] != grid_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', the first dimension of 'grid' and 'input_x' must be equal, but got the shape of 'grid' is "
        << input_args[kInputIndex1]->GetShape()->ToString() << " , and the shape of 'input_x' is "
        << input_args[kInputIndex0]->GetShape()->ToString() << ".";
    }
    if (grid_shape[kInputIndex4] != static_cast<int64_t>(kInputIndex3)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the last dimension of grid must be 3, but got "
                               << std::to_string(grid_shape[kInputIndex4]) << ".";
    }
  }
  std::vector<int64_t> output_shape = {input_x_shape[kInputIndex0], input_x_shape[kInputIndex1],
                                       grid_shape[kInputIndex1], grid_shape[kInputIndex2], grid_shape[kInputIndex3]};
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr GridSampler3DFuncImpl::InferType(const PrimitivePtr &prim,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_x_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_x_type);
  auto grid_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_x_type);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_x", input_x_type);
  (void)types.emplace("grid", grid_type);
  (void)CheckAndConvertUtils::CheckTypeSame(types, prim->name());
  return input_x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
