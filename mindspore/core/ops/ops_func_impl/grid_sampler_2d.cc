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

#include "ops/ops_func_impl/grid_sampler_2d.h"

#include <algorithm>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr GridSampler2DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto input_x_base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input_x_base_shape);
  auto input_x_shape = input_x_base_shape->GetShapeVector();

  auto grid_base_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(grid_base_shape);
  auto grid_shape = grid_base_shape->GetShapeVector();

  const int64_t normal_shape_size = 4;
  const int64_t label3 = 3;
  const int64_t num2 = 2;
  // dynamic rank
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape)) {
    return std::make_shared<abstract::TensorShape>(
      ShapeVector{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                  abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny});
  }
  // dynamic shape
  if (IsDynamicRank(input_x_shape)) {
    input_x_shape = {grid_shape[0], abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                     abstract::TensorShape::kShapeDimAny};
  }
  if (IsDynamicRank(grid_shape)) {
    grid_shape = {input_x_shape[0], abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny, 2};
  }
  if (input_x_shape.size() != normal_shape_size) {
    MS_EXCEPTION(ValueError) << "Input_x must be a 4-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != normal_shape_size) {
    MS_EXCEPTION(ValueError) << "Grid must be a 4-dimensional tensor, but got " << std::to_string(grid_shape.size())
                             << "-dimensional tensor.";
  }
  if (!IsDynamic(input_x_shape) && !IsDynamic(grid_shape)) {
    if (input_x_shape[0] != grid_shape[0]) {
      MS_EXCEPTION(ValueError) << "The shape of grid is " << input_args[1]->GetShape()->ToString()
                               << " , but the shape of input_x is " << input_args[0]->GetShape()->ToString()
                               << " . The first dimension of grid and input_x must be equal.";
    }
    if (grid_shape[label3] != num2) {
      MS_EXCEPTION(ValueError) << "The forth dimension of grid must be 2, but got "
                               << std::to_string(grid_shape[label3]);
    }
  }
  std::vector<int64_t> output_shape = {input_x_shape[0], input_x_shape[1], grid_shape[1], grid_shape[2]};
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr GridSampler2DFuncImpl::InferType(const PrimitivePtr &prim,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_x_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_x_type);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_grid_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_grid_type);
  if (input_x_type->ToString() != input_grid_type->ToString()) {
    MS_EXCEPTION(TypeError) << "Input grid must have the same data type with input x! input[x] data type = "
                            << input_x_type->ToString()
                            << " but input[grid] data type = " << input_grid_type->ToString();
  }
  return input_x_type->Clone();
}

}  // namespace ops
}  // namespace mindspore
