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
#include "ops/ops_func_impl/stack.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;
int64_t get_axis(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  int64_t axis_temp;
  if (primitive->HasAttr(kAxis)) {
    axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  } else {
    axis_temp = GetValue<int64_t>(input_args[1]->BuildValue());
  }
  return axis_temp;
}
}  // namespace

BaseShapePtr StackFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  AbstractBasePtrList elements = input_args;
  if (input_args[0]->isa<abstract::AbstractSequence>()) {
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  bool has_rank_valid_shape = false;
  ShapeVector input_shape;
  size_t element_rank = 0;
  for (size_t i = 0; i < elements.size(); ++i) {
    MS_EXCEPTION_IF_NULL(elements[i]);
    auto input_shape_tmp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->GetShape())[kShape];
    if (IsDynamicRank(input_shape_tmp)) {
      continue;
    }
    if (!has_rank_valid_shape) {
      has_rank_valid_shape = true;
      input_shape = input_shape_tmp;
      element_rank = input_shape_tmp.size();
      continue;
    }
    if (input_shape_tmp.size() != input_shape.size()) {
      MS_EXCEPTION(ValueError) << "All input shape size must be the same!";
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (input_shape.at(j) == kUnknownDim && input_shape_tmp.at(j) != kUnknownDim) {
        input_shape[j] = input_shape_tmp.at(j);
        continue;
      }
      if (input_shape_tmp.at(j) != input_shape.at(j)) {
        MS_EXCEPTION(ValueError) << "All input shape must be the same! " << input_shape_tmp << " And " << input_shape;
      }
    }
  }
  if (!has_rank_valid_shape) {
    return std::make_shared<abstract::Shape>(ShapeVector{kUnknownRank});
  }
  std::vector<int64_t> infer_shape = input_shape;
  int64_t axis_temp = get_axis(primitive, input_args);
  CheckAndConvertUtils::CheckInRange<int64_t>("Stack axis", axis_temp, kIncludeBoth,
                                              {-SizeToLong(element_rank) - 1, SizeToLong(element_rank)},
                                              primitive->name());
  auto axis = axis_temp < 0 ? static_cast<size_t>(axis_temp) + element_rank + 1 : LongToSize(axis_temp);
  (void)infer_shape.insert(infer_shape.begin() + axis, elements.size());
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr StackFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  AbstractBasePtrList elements = input_args;
  if (input_args[0]->isa<abstract::AbstractSequence>()) {
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  primitive->AddAttr("num", MakeValue(SizeToLong(elements.size())));
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto infer_type0 = element0->GetType();
  return infer_type0;
}
}  // namespace ops
}  // namespace mindspore
