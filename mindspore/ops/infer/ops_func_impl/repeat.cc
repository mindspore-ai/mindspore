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

#include <vector>
#include "infer/ops_func_impl/repeat.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/op_name.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
namespace {
inline ShapeValueDType InferRepeatCheckRepeatValue(const int64_t dim, ShapeValueDType dim_size) {
  if (dim_size < 0) {
    MS_EXCEPTION(RuntimeError) << "For Repeat, output size of all dimensions should be no smaller than 0, but got "
                               << dim_size << " at dim " << dim << " .";
  }
  return dim_size;
}

ShapeArray InferRepeatWithRepeatsValueKnowen(const ShapeVector &input_shape,
                                             const ArrayValue<ShapeValueDType> &repeats) {
  size_t repeat_it = 0;
  ShapeVector output_shape{};
  const auto repeat_len = repeats.size();
  const auto input_rank = input_shape.size();
  auto copy_count = repeat_len - input_rank;
  for (; repeat_it < copy_count; ++repeat_it) {
    output_shape.push_back(InferRepeatCheckRepeatValue(repeat_it, repeats[repeat_it]));
  }
  for (auto shape_it : input_shape) {
    if (shape_it == abstract::Shape::kShapeDimAny) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      output_shape.push_back(InferRepeatCheckRepeatValue(repeat_it, repeats[repeat_it] * shape_it));
    }
    ++repeat_it;
  }
  return {output_shape};
}
}  // namespace

ShapeArray RepeatFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto repeat = input_infos[kIndex1]->GetArrayValue<ShapeValueDType>();
  // Either rank of self or value of repeat is unknown
  if (input_infos[kIndex0]->IsDynamicRank() || !repeat.has_value()) {
    return {{abstract::Shape::kShapeRankAny}};
  }
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto repeat_value = repeat.value();
  auto input_rank = input_shape.size();
  if (input_rank > repeat_value.size()) {
    MS_EXCEPTION(ValueError) << "For repeat, number of items of repeats can not be smaller than the number of "
                                "dimensions of self tensor, but got repeats with "
                             << repeat_value.size() << " items and rank of self Tensor is " << input_rank << ".";
  }
  // Quick path without unknownValue
  if (!repeat_value.HasUnknownValue()) {
    return InferRepeatWithRepeatsValueKnowen(input_shape, repeat_value);
  }
  ShapeVector output_shape{};
  size_t repeat_it = 0;
  for (; repeat_it < (repeat_value.size() - input_rank); ++repeat_it) {
    if (repeat_value.IsValueUnknown(repeat_it)) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      output_shape.push_back(InferRepeatCheckRepeatValue(repeat_it, repeat_value[repeat_it]));
    }
  }
  for (auto shape_it : input_shape) {
    if (repeat_value.IsValueUnknown(repeat_it) || (shape_it == abstract::Shape::kShapeDimAny)) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      output_shape.push_back(InferRepeatCheckRepeatValue(repeat_it, repeat_value[repeat_it] * shape_it));
    }
    ++repeat_it;
  }
  return {output_shape};
}

std::vector<TypeId> RepeatFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
