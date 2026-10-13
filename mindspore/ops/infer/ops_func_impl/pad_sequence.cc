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

#include "infer/ops_func_impl/pad_sequence.h"

#include <algorithm>
#include <string>
#include <vector>

#include "utils/shape_utils.h"

namespace mindspore::ops {
namespace {
bool UpdateMaxSequenceLength(const PrimitivePtr &primitive, const ShapeVector &first_shape, const ShapeVector &current,
                             int64_t *max_length) {
  if (IsDynamicRank(current)) {
    return false;
  }
  if (current.size() != first_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', sequence tensors must have the same rank.";
  }
  for (size_t dim = 1; dim < first_shape.size(); ++dim) {
    if (first_shape[dim] != abstract::TensorShape::kShapeDimAny &&
        current[dim] != abstract::TensorShape::kShapeDimAny && first_shape[dim] != current[dim]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', trailing dimensions must match.";
    }
  }
  if (*max_length == abstract::TensorShape::kShapeDimAny || current[0] == abstract::TensorShape::kShapeDimAny) {
    *max_length = abstract::TensorShape::kShapeDimAny;
  } else {
    *max_length = std::max(*max_length, current[0]);
  }
  return true;
}
}  // namespace

ShapeArray PadSequenceFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &sequence = input_infos[0];
  if (sequence->IsDynamicSequence()) {
    const auto element_shape = sequence->GetDynamicSequenceElement()->GetShape();
    if (IsDynamicRank(element_shape)) {
      return {{abstract::TensorShape::kShapeRankAny}};
    }
    ShapeVector output(element_shape.size() + 1, abstract::TensorShape::kShapeDimAny);
    std::copy(element_shape.begin() + 1, element_shape.end(), output.begin() + 2);
    return {output};
  }

  const auto &elements = sequence->GetSequenceElements();
  if (elements.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', sequences must not be empty.";
  }
  const auto first_shape = elements.front()->GetShape();
  if (IsDynamicRank(first_shape)) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  if (first_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', every sequence tensor must have rank >= 1.";
  }

  int64_t max_length = first_shape[0];
  for (size_t i = 1; i < elements.size(); ++i) {
    const auto current = elements[i]->GetShape();
    if (!UpdateMaxSequenceLength(primitive, first_shape, current, &max_length)) {
      return {{abstract::TensorShape::kShapeRankAny}};
    }
  }

  const auto padding_side = input_infos[3]->GetScalarValue<int64_t>();
  if (padding_side.has_value() && padding_side.value() != 0 && padding_side.value() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', padding_side must be 'left' or 'right'.";
  }
  const auto batch_first = input_infos[1]->GetScalarValue<bool>();
  const int64_t batch_size = SizeToLong(elements.size());
  ShapeVector output;
  if (!batch_first.has_value()) {
    output = {abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
  } else if (batch_first.value()) {
    output = {batch_size, max_length};
  } else {
    output = {max_length, batch_size};
  }
  output.insert(output.end(), first_shape.begin() + 1, first_shape.end());
  return {output};
}

std::vector<TypeId> PadSequenceFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  const auto &sequence = input_infos[0];
  if (sequence->IsDynamicSequence()) {
    return {sequence->GetDynamicSequenceElement()->GetType()};
  }
  const auto &elements = sequence->GetSequenceElements();
  if (elements.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', sequences must not be empty.";
  }
  const auto output_type = elements.front()->GetType();
  for (const auto &element : elements) {
    if (element->GetType() != output_type) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', all sequence tensors must have the same dtype.";
    }
  }
  return {output_type};
}
}  // namespace mindspore::ops
