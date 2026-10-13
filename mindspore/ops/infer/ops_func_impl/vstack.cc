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

#include "infer/ops_func_impl/vstack.h"

#include <algorithm>
#include <vector>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore::ops {
namespace {
ShapeVector NormalizeVstackShape(const ShapeVector &shape) {
  if (IsDynamicRank(shape)) {
    return shape;
  }
  if (shape.empty()) {
    return {1, 1};
  }
  if (shape.size() == 1) {
    return {1, shape[0]};
  }
  return shape;
}

TypeId InferVstackType(const PrimitivePtr &primitive, const InferInfoPtr &sequence) {
  if (sequence->IsDynamicSequence()) {
    return sequence->GetDynamicSequenceElement()->GetType();
  }
  const auto &elements = sequence->GetSequenceElements();
  if (elements.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', tensors must not be empty.";
  }
  auto output_type = TypeIdToType(elements.front()->GetType());
  for (const auto &element : elements) {
    output_type = PromoteType(output_type, TypeIdToType(element->GetType()), primitive->name());
  }
  return output_type->type_id();
}
}  // namespace

ShapeArray VstackFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &sequence = input_infos[0];
  if (sequence->IsDynamicSequence()) {
    auto output = NormalizeVstackShape(sequence->GetDynamicSequenceElement()->GetShape());
    if (!IsDynamicRank(output)) {
      output[0] = abstract::TensorShape::kShapeDimAny;
    }
    return {output};
  }
  const auto &elements = sequence->GetSequenceElements();
  if (elements.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', tensors must not be empty.";
  }
  auto output = NormalizeVstackShape(elements.front()->GetShape());
  if (IsDynamicRank(output)) {
    return {output};
  }
  for (size_t i = 1; i < elements.size(); ++i) {
    const auto current = NormalizeVstackShape(elements[i]->GetShape());
    if (IsDynamicRank(current)) {
      return {{abstract::TensorShape::kShapeRankAny}};
    }
    if (current.size() != output.size()) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', all tensors must have the same rank after "
                               << "atleast_2d normalization.";
    }
    for (size_t dim = 1; dim < output.size(); ++dim) {
      if (output[dim] != abstract::TensorShape::kShapeDimAny && current[dim] != abstract::TensorShape::kShapeDimAny &&
          output[dim] != current[dim]) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', tensor shapes are incompatible.";
      }
      if (current[dim] == abstract::TensorShape::kShapeDimAny) {
        output[dim] = abstract::TensorShape::kShapeDimAny;
      }
    }
    if (output[0] == abstract::TensorShape::kShapeDimAny || current[0] == abstract::TensorShape::kShapeDimAny) {
      output[0] = abstract::TensorShape::kShapeDimAny;
    } else {
      output[0] += current[0];
    }
  }
  return {output};
}

std::vector<TypeId> VstackFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  return {InferVstackType(primitive, input_infos[0])};
}
}  // namespace mindspore::ops
