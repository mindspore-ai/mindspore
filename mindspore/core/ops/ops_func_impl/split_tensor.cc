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
int64_t CaculateSplitSections(const AbstractBasePtr &input_abs) {
  auto split_size_value = input_abs->GetValue();
  auto split_size = GetScalarValue<int64_t>(split_size_value);
  if (MS_UNLIKELY(!split_size.has_value())) {
    MS_LOG(EXCEPTION) << "split_size's value is valueany";
  }
  if (split_size.value() == 0) {
    MS_EXCEPTION(ValueError) << "split_size's value cannot be zero";
  }
  return split_size.value();
}
}  // namespace
BaseShapePtr SplitTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto axis_value = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  if (MS_UNLIKELY(!axis_value.has_value())) {
    MS_LOG(EXCEPTION) << "split_size's value is valueany";
  }

  auto axis = axis_value.value();
  std::vector<abstract::BaseShapePtr> output_list;
  auto split_sections = CaculateSplitSections(input_args[kIndex1]);
  auto rank = SizeToLong(input_shape.size());

  if (axis < 0) {
    axis += rank;
  }
  size_t pos = LongToSize(axis);

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
    output_list.push_back(last_output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TypePtr SplitTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto axis_value = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  if (MS_UNLIKELY(!axis_value.has_value())) {
    MS_LOG(EXCEPTION) << "axis's value is valueany";
  }

  auto axis = axis_value.value();
  auto rank = SizeToLong(input_shape.size());
  if (axis < 0) {
    axis += rank;
  }
  size_t pos = LongToSize(axis);
  auto split_sections = CaculateSplitSections(input_args[kIndex1]);
  auto output_num = (input_shape[pos] % split_sections) == 0 ? (input_shape[pos] / split_sections)
                                                             : (input_shape[pos] / split_sections) + 1;
  auto infer_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  static const std::set<TypePtr> valid_types = {kInt8,    kInt16,   kInt32,     kInt64,      kFloat16,
                                                kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, primitive->name());
  std::vector<TypePtr> type_tuple;
  for (int32_t i = 0; i < output_num; i++) {
    (void)type_tuple.emplace_back(type->Clone());
  }
  return std::make_shared<Tuple>(type_tuple);
}

int32_t SplitTensorFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  int32_t check_status = OP_CHECK_SUCCESS;
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", rank, kGreaterEqual, 1, primitive));
  auto axis_value = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  if (MS_UNLIKELY(!axis_value.has_value()) || IsDynamicRank(input_shape)) {
    return OP_CHECK_RETRY;
  }
  auto axis = axis_value.value();
  MS_CHECK_VALUE(axis >= -rank && axis < rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis, kIncludeLeft, {-rank, rank}, primitive));
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.value() <= 0) {
      check_status = OP_CHECK_RETRY;
    }
  }
  return check_status;
}

}  // namespace ops
}  // namespace mindspore
