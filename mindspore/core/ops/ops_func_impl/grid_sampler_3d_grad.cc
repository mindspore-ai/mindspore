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

#include "ops/ops_func_impl/grid_sampler_3d_grad.h"

#include <algorithm>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {

const size_t k5DSize = 5;

BaseShapePtr GridSampler3DGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto grad_base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(grad_base_shape);
  const auto &grad_shape = grad_base_shape->GetShapeVector();

  auto input_x_base_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input_x_base_shape);
  auto input_x_shape = input_x_base_shape->GetShapeVector();

  auto grid_base_shape = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(grid_base_shape);
  auto grid_shape = grid_base_shape->GetShapeVector();
  if (IsDynamicRank(input_x_shape) || IsDynamicRank(grid_shape)) {
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{
      std::make_shared<abstract::TensorShape>(ShapeVector{
        abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
        abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny}),
      std::make_shared<abstract::TensorShape>(
        ShapeVector{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                    abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny, 3})});
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
  if (grad_shape.size() != k5DSize) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'grad' must be a 5-D tensor, but got a "
                             << std::to_string(grad_shape.size()) << "-D tensor.";
  }
  if (input_x_shape.size() != k5DSize) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'input_x' must be a 5-D tensor, but got a "
                             << std::to_string(input_x_shape.size()) << "-D tensor.";
  }
  if (grid_shape.size() != k5DSize) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'grid' must be a 5-D tensor, but got a "
                             << std::to_string(grid_shape.size()) << "-D tensor.";
  }
  if (!IsDynamic(input_x_shape) && !IsDynamic(grid_shape)) {
    if (input_x_shape[kInputIndex0] != grid_shape[kInputIndex0]) {
      MS_EXCEPTION(ValueError)
        << "For '" << primitive->name()
        << "', the first dimension of 'grid' and 'input_x' must be equal. But got the shape of 'grid': "
        << input_args[kInputIndex2]->GetShape()->ToString()
        << ", the shape of 'input_x': " << input_args[kInputIndex1]->GetShape()->ToString() << ".";
    }
    if (grid_shape[kInputIndex4] != SizeToLong(kInputIndex3)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the last dimension of 'grid' must be 3, but got "
                               << std::to_string(grid_shape[kInputIndex4]) << ".";
    }
  }
  std::vector<int64_t> out_shape = {input_x_shape[kInputIndex0], input_x_shape[kInputIndex1], grid_shape[kInputIndex1],
                                    grid_shape[kInputIndex2], grid_shape[kInputIndex3]};
  if (!IsDynamic(out_shape) && !IsDynamic(grad_shape)) {
    bool shape_error = false;
    for (size_t i = kInputIndex0; i < k5DSize; i++) {
      if (out_shape[i] != grad_shape[i]) {
        shape_error = true;
        break;
      }
    }
    if (shape_error) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the shape of 'grad' must be the same as that of output, but got 'grad' shape: "
                               << input_args[kInputIndex0]->GetShape()->ToString() << ", output shape: ("
                               << std::to_string(out_shape[kInputIndex0]) << ", "
                               << std::to_string(out_shape[kInputIndex1]) << ", "
                               << std::to_string(out_shape[kInputIndex2]) << ", "
                               << std::to_string(out_shape[kInputIndex3]) << ", "
                               << std::to_string(out_shape[kInputIndex4]) << ").";
    }
  }
  abstract::TensorShapePtr dx_shape = std::make_shared<abstract::TensorShape>(input_x_shape);
  abstract::TensorShapePtr dgrid_shape = std::make_shared<abstract::TensorShape>(grid_shape);
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{dx_shape, dgrid_shape});
}
TypePtr GridSampler3DGradFuncImpl::InferType(const PrimitivePtr &prim,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_x_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_x_type);

  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  auto grid_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(grid_type);
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_x_type->Clone(), grid_type->Clone()});
}
}  // namespace ops
}  // namespace mindspore
