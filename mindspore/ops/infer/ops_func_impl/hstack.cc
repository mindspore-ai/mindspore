/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/hstack.h"

#include <vector>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
namespace {
constexpr auto kHstackOpName = "Hstack";
ShapeVector NormalizeHstackShape(const ShapeVector &shape) {
  if (IsDynamicRank(shape)) {
    return shape;
  }
  return shape.empty() ? ShapeVector{1} : shape;
}

ShapeVector InferHstackShape(const PrimitivePtr &primitive, const ShapeArray &shapes) {
  if (shapes.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', tensors must not be empty.";
  }
  auto output = NormalizeHstackShape(shapes.front());
  if (IsDynamicRank(output)) {
    return output;
  }
  const size_t axis = output.size() == 1 ? 0 : 1;
  for (size_t i = 1; i < shapes.size(); ++i) {
    const auto current = NormalizeHstackShape(shapes[i]);
    if (IsDynamicRank(current)) {
      return {abstract::TensorShape::kShapeRankAny};
    }
    if (current.size() != output.size()) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', all tensors must have the same rank after "
                               << "atleast_1d normalization.";
    }
    for (size_t dim = 0; dim < output.size(); ++dim) {
      if (dim == axis) {
        continue;
      }
      if (output[dim] != abstract::TensorShape::kShapeDimAny && current[dim] != abstract::TensorShape::kShapeDimAny &&
          output[dim] != current[dim]) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', tensor shapes are incompatible.";
      }
      if (current[dim] == abstract::TensorShape::kShapeDimAny) {
        output[dim] = abstract::TensorShape::kShapeDimAny;
      }
    }
    if (output[axis] == abstract::TensorShape::kShapeDimAny || current[axis] == abstract::TensorShape::kShapeDimAny) {
      output[axis] = abstract::TensorShape::kShapeDimAny;
    } else {
      output[axis] += current[axis];
    }
  }
  return output;
}

}  // namespace

ShapeArray HstackFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto tensors = input_values[0]->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(tensors);
  ShapeArray shapes;
  for (const auto &value : tensors->value()) {
    const auto input_tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    shapes.push_back(input_tensor->shape());
  }
  return {InferHstackShape(primitive, shapes)};
}

TypePtrList HstackFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto tensors = input_values[0]->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(tensors);
  if (tensors->size() == 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', tensors must not be empty.";
  }
  TypePtr output_type = nullptr;
  for (const auto &value : tensors->value()) {
    const auto input_tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    output_type = output_type == nullptr ? input_tensor->Dtype()
                                         : PromoteType(output_type, input_tensor->Dtype(), primitive->name());
  }
  return {output_type};
}

REGISTER_SIMPLE_INFER(kHstackOpName, HstackFuncImpl);
}  // namespace mindspore::ops
