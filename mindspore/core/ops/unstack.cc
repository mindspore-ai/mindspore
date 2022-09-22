/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/unstack.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr UnstackInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());

  auto output_num = GetValue<int64_t>(primitive->GetAttr("num"));
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  auto axis = axis_temp < 0 ? LongToSize(axis_temp + x_rank) : LongToSize(axis_temp);

  auto temp_shape = x_shape;
  (void)temp_shape.erase(temp_shape.begin() + axis);
  std::vector<abstract::BaseShapePtr> shape_tuple;
  for (int64_t i = 0; i < output_num; ++i) {
    abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(temp_shape);
    shape_tuple.push_back(out_shape);
  }
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TuplePtr UnstackInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  auto type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", type, common_valid_types_with_complex_and_bool, name);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("x_rank", x_rank, kGreaterEqual, 1, name);

  auto axis_temp = GetValue<int64_t>(prim->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange("axis value", axis_temp, kIncludeLeft, {-x_rank, x_rank}, name);
  auto axis = axis_temp < 0 ? LongToSize(axis_temp + x_rank) : LongToSize(axis_temp);

  auto output_num = x_shape[axis];
  (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterEqual, 1, name);
  (void)prim->AddAttr("num", MakeValue(output_num));

  std::vector<TypePtr> type_tuple;
  for (int64_t i = 0; i < output_num; ++i) {
    type_tuple.push_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Unstack, BaseOperator);
AbstractBasePtr UnstackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infertype = UnstackInferType(primitive, input_args);
  auto infershape = UnstackInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
void Unstack::Init(const int64_t axis) { this->set_axis(axis); }
void Unstack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t Unstack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
REGISTER_PRIMITIVE_EVAL_IMPL(Unstack, prim::kPrimUnstack, UnstackInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
