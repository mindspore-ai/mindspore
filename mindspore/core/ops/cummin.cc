/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <set>
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "ops/cummin.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr CumminInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  constexpr int64_t min_dim = 0;
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", SizeToLong(x_shape.size()), kGreaterThan, min_dim, prim_name);
  int64_t axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  CheckAndConvertUtils::CheckInRange<int64_t>(kAxis, axis, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape_ptr, x_shape_ptr});
}

TuplePtr CumminInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = common_valid_types;
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  TypePtr indices_type = kInt32;
  return std::make_shared<Tuple>(std::vector{x_type, indices_type});
}
}  // namespace

void Cummin::Init(const int64_t &axis) { this->set_axis(axis); }

void Cummin::set_axis(const int64_t &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

int64_t Cummin::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
MIND_API_OPERATOR_IMPL(Cummin, BaseOperator);

AbstractBasePtr CumminInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = CumminInferType(primitive, input_args);
  auto shape = CumminInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Cummin, prim::kPrimCummin, CumminInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
