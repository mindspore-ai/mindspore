/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/split.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Split, BaseOperator);
void Split::Init(const int64_t axis, const int64_t output_num) {
  this->set_axis(axis);
  this->set_output_num(output_num);
}

void Split::set_size_splits(const std::vector<int64_t> &size_splits) {
  (void)this->AddAttr(kSizeSplits, api::MakeValue(size_splits));
}
void Split::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
void Split::set_output_num(const int64_t output_num) { (void)this->AddAttr(kOutputNum, api::MakeValue(output_num)); }

std::vector<int64_t> Split::get_size_splits() const {
  auto value_ptr = GetAttr(kSizeSplits);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Split::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

int64_t Split::get_output_num() const {
  auto value_ptr = GetAttr(kOutputNum);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::TupleShapePtr SplitInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, 1L, prim_name);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();
  auto input_min_shape = shape_ptr->min_shape();
  auto input_max_shape = shape_ptr->max_shape();

  auto input_rank = SizeToLong(input_shape.size());
  auto output_num = GetValue<int64_t>(primitive->GetAttr(kOutputNum));
  auto axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  (void)CheckAndConvertUtils::CheckInteger("input_rank", input_rank, kGreaterEqual, 1, prim_name);

  ShapeVector out_shape = input_shape;
  ShapeVector out_min_shape = input_min_shape;
  ShapeVector out_max_shape = input_max_shape;
  if (!shape_ptr->IsDimUnknown()) {
    (void)CheckAndConvertUtils::CheckInteger(kAxis, axis, kLessThan, input_rank, prim_name);
    axis = axis < 0 ? axis + input_rank : axis;
    int64_t split_axis_length = -1;
    if (input_shape[axis] != -1) {
      (void)CheckAndConvertUtils::CheckInteger("input x", input_shape[axis] % output_num, kEqual, 0L, prim_name);
      split_axis_length = input_shape[axis] / output_num;
    }
    out_shape[axis] = split_axis_length;
  }
  std::vector<abstract::BaseShapePtr> shape_tuple;
  for (int64_t i = 0; i < output_num; i++) {
    abstract::ShapePtr output_shape = std::make_shared<abstract::Shape>(out_shape, out_min_shape, out_max_shape);
    shape_tuple.push_back(output_shape);
  }
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TuplePtr SplitInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto output_num = GetValue<int64_t>(prim->GetAttr(kOutputNum));
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,  kInt64,   kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat16, kFloat32};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input_x", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  for (int64_t i = 0; i < output_num; i++) {
    type_tuple.push_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

AbstractBasePtr SplitInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infertype = SplitInferType(primitive, input_args);
  auto infershape = SplitInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

REGISTER_PRIMITIVE_C(kNameSplit, Split);
}  // namespace ops
}  // namespace mindspore
