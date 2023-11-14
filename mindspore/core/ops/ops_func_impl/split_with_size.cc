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
#include "ops/ops_func_impl/split_with_size.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CaculateAxis(const AbstractBasePtr &input_abs) {
  auto axis_value = input_abs->GetValue();
  if (axis_value == nullptr || axis_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitWithSize op, axis should be int64_t, but got " << axis_value->ToString();
  }
  auto axis = GetValue<int64_t>(axis_value);
  return axis;
}

std::vector<int64_t> CaculateSplitSize(const AbstractBasePtr &input_abs) {
  auto split_size_value = input_abs->GetValue()->cast<ValueTuplePtr>();
  if (split_size_value == nullptr || split_size_value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << "For SplitWithSize op, split size should be tuple[int], but got "
                      << split_size_value->ToString();
  }
  std::vector<int64_t> split_size = std::move(GetValue<std::vector<int64_t>>(split_size_value));
  return split_size;
}
}  // namespace
BaseShapePtr SplitWithSizeFuncImpl::InferShape(const PrimitivePtr &primitive,
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

  auto split_size = CaculateSplitSize(input_args[kIndex1]);
  int64_t sum_split_size = std::accumulate(split_size.begin(), split_size.end(), 0);
  (void)CheckAndConvertUtils::CheckInteger("sum_split_size", sum_split_size, kEqual, input_shape[pos], prim_name);
  auto output_shape = input_shape;
  for (const int64_t &size : split_size) {
    output_shape[pos] = size;
    abstract::ShapePtr output = std::make_shared<abstract::Shape>(output_shape);
    (void)output_list.emplace_back(output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TypePtr SplitWithSizeFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto &prim_name = primitive->name();
  auto split_size = CaculateSplitSize(input_args[kIndex1]);
  auto infer_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  static const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, prim_name);
  std::vector<TypePtr> type_tuple;
  for (size_t i = 0; i < split_size.size(); i++) {
    (void)type_tuple.emplace_back(type->Clone());
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace ops
}  // namespace mindspore
