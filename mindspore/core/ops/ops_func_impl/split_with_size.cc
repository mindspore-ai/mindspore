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
#include <utility>
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> CaculateSplitSize(const AbstractBasePtr &input_abs) {
  std::vector<int64_t> split_size = GetArrayValue<int64_t>(input_abs).value().ToVector();
  return split_size;
}
}  // namespace
BaseShapePtr SplitWithSizeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto axis_value = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  if (MS_UNLIKELY(!axis_value.has_value())) {
    MS_LOG(EXCEPTION) << "axis's value is valueany";
  }
  auto axis = axis_value.value();
  std::vector<abstract::BaseShapePtr> output_list;

  auto rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", rank, kGreaterEqual, 1, primitive));
  MS_CHECK_VALUE(axis >= -rank && axis < rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis, kIncludeLeft, {-rank, rank}, primitive));
  if (axis < 0) {
    axis += rank;
  }
  size_t pos = LongToSize(axis);
  auto split_size = CaculateSplitSize(input_args[kIndex1]);
  int64_t sum_split_size = std::accumulate(split_size.begin(), split_size.end(), 0);
  MS_CHECK_VALUE(sum_split_size == input_shape[pos],
                 CheckAndConvertUtils::FormatCheckIntegerMsg("sum_split_size", sum_split_size, kEqual,
                                                             SizeToLong(input_shape[pos]), primitive));

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
  static const std::set<TypePtr> valid_types = {kInt8,    kInt16,   kInt32,     kInt64,      kFloat16,
                                                kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, prim_name);
  std::vector<TypePtr> type_tuple;
  for (size_t i = 0; i < split_size.size(); i++) {
    (void)type_tuple.emplace_back(type->Clone());
  }
  return std::make_shared<Tuple>(type_tuple);
}

}  // namespace ops
}  // namespace mindspore
