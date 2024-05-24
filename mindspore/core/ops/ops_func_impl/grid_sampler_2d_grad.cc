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

#include "ops/ops_func_impl/grid_sampler_2d_grad.h"

#include <algorithm>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {

BaseShapePtr GridSampler2DGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto grad_base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(grad_base_shape);
  auto grad_shape = grad_base_shape->GetShapeVector();

  auto input_x_base_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input_x_base_shape);
  auto input_x_shape = input_x_base_shape->GetShapeVector();

  auto grid_base_shape = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(grid_base_shape);
  auto grid_shape = grid_base_shape->GetShapeVector();
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape)) {
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{
      std::make_shared<abstract::TensorShape>(
        ShapeVector{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                    abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny}),
      std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeDimAny,
                                                          abstract::TensorShape::kShapeDimAny,
                                                          abstract::TensorShape::kShapeDimAny, 2})});
  }
  // dynamic shape
  if (IsDynamicRank(input_x_shape)) {
    input_x_shape = {grid_shape[0], -1, -1, -1};
  }
  if (IsDynamicRank(grid_shape)) {
    grid_shape = {input_x_shape[0], -1, -1, 2};
  }
  if (grad_shape.size() != kInputIndex4) {
    MS_EXCEPTION(ValueError) << "Grad must be a 4-dimensional tensor, but got " << std::to_string(grad_shape.size())
                             << "-dimensional tensor.";
  }
  if (input_x_shape.size() != kInputIndex4) {
    MS_EXCEPTION(ValueError) << "Input_x must be a 4-dimensional tensor, but got "
                             << std::to_string(input_x_shape.size()) << "-dimensional tensor.";
  }
  if (grid_shape.size() != kInputIndex4) {
    MS_EXCEPTION(ValueError) << "Grid must be a 4-dimensional tensor, but got " << std::to_string(grid_shape.size())
                             << "-dimensional tensor.";
  }
  if (!IsDynamic(input_x_shape) && !IsDynamic(grid_shape)) {
    if (input_x_shape[kInputIndex0] != grid_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError) << "The shape of grid is " << input_args[kInputIndex2]->GetShape()->ToString()
                               << " , but the shape of input_x is " << input_args[kInputIndex1]->GetShape()->ToString()
                               << " . The first dimension of grid and input_x must be equal.";
    }
    if (grid_shape[kInputIndex3] != SizeToLong(kInputIndex2)) {
      MS_EXCEPTION(ValueError) << "The last dimension of grid must be 2, but got "
                               << std::to_string(grid_shape[kInputIndex3]);
    }
  }
  std::vector<int64_t> out_shape = {input_x_shape[kInputIndex0], input_x_shape[kInputIndex1], grid_shape[kInputIndex1],
                                    grid_shape[kInputIndex2]};
  if (!IsDynamic(out_shape) && !IsDynamic(grad_shape)) {
    bool shape_error = false;
    for (size_t i = kInputIndex0; i < kInputIndex4; i++) {
      if (out_shape[i] != grad_shape[i]) {
        shape_error = true;
        break;
      }
    }
    if (shape_error) {
      MS_EXCEPTION(ValueError) << "The shape of grad, which is the same as that of output, is "
                               << input_args[kInputIndex0]->GetShape()->ToString() << ", but the shape of output is ("
                               << std::to_string(out_shape[kInputIndex0]) << ", "
                               << std::to_string(out_shape[kInputIndex1]) << ", "
                               << std::to_string(out_shape[kInputIndex2]) << ", "
                               << std::to_string(out_shape[kInputIndex3]) << ").";
    }
  }
  auto dx_shape = std::make_shared<abstract::TensorShape>(input_x_shape);
  auto dgrid_shape = std::make_shared<abstract::TensorShape>(grid_shape);
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{dx_shape, dgrid_shape});
}

TypePtr GridSampler2DGradFuncImpl::InferType(const PrimitivePtr &prim,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  auto input_x_type = input_args[kIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_x_type);

  MS_EXCEPTION_IF_NULL(input_args[kIndex2]);
  auto grid_type = input_args[kIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(grid_type);
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_x_type->Clone(), grid_type->Clone()});
}

}  // namespace ops
}  // namespace mindspore
