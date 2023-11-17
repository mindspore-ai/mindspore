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
#include "ops/ops_func_impl/split_tensor.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CaculateAxis(const AbstractBasePtr &input_abs) {
  auto axis_value = input_abs->GetValue();
  if (axis_value == nullptr || axis_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitTensor op, axis should be int64_t, but got " << axis_value->ToString();
  }
  auto axis = GetValue<int64_t>(axis_value);
  return axis;
}

int64_t CaculateSplitSections(const AbstractBasePtr &input_abs) {
  auto split_size_value = input_abs->GetValue();
  if (split_size_value == nullptr || split_size_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitTensor op, split sections should be int64_t, but got "
                      << split_size_value->ToString();
  }
  auto split_size = GetValue<int64_t>(split_size_value);
  return split_size;
}
}  // namespace
BaseShapePtr SplitTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto axis = CaculateAxis(input_args[kIndex2]);
  size_t pos = LongToSize(axis);
  std::vector<abstract::BaseShapePtr> output_list;

  auto rank = SizeToLong(input_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank", rank, kGreaterEqual, 1, prim_name);
  if (axis < 0) {
    axis += rank;
  }
  CheckAndConvertUtils::CheckInRange("axis", axis, kIncludeLeft, {-rank, rank}, prim_name);

  auto split_sections = CaculateSplitSections(input_args[kIndex1]);
  auto output_shape = input_shape;
  output_shape[pos] = split_sections;
  for (int64_t i = 0; i < input_shape[pos] / split_sections; ++i) {
    abstract::ShapePtr output = std::make_shared<abstract::Shape>(output_shape);
    (void)output_list.emplace_back(output);
  }
  int64_t last_size = input_shape[pos] % split_sections;
  if (last_size != 0) {
    auto last_shape = input_shape;
    last_shape[pos] = last_size;
    abstract::ShapePtr last_output = std::make_shared<abstract::Shape>(last_shape);
    (void)output_list.push_back(last_output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TypePtr SplitTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto axis = CaculateAxis(input_args[kIndex2]);
  size_t pos = LongToSize(axis);

  auto split_sections = CaculateSplitSections(input_args[kIndex1]);
  auto output_num = (input_shape[pos] % split_sections) == 0 ? (input_shape[pos] / split_sections)
                                                             : (input_shape[pos] / split_sections) + 1;
  auto infer_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  static const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, primitive->name());
  std::vector<TypePtr> type_tuple;
  for (int32_t i = 0; i < output_num; i++) {
    (void)type_tuple.emplace_back(type->Clone());
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace ops
}  // namespace mindspore
