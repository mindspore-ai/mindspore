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
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kAttrNum = "num";
constexpr int64_t kUnstackInputsNum = 1;
}  // namespace
MIND_API_OPERATOR_IMPL(Unstack, BaseOperator);

void Unstack::Init(const int64_t axis) { this->set_axis(axis); }
void Unstack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t Unstack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

namespace {
size_t GetUnstackAxis(const std::vector<int64_t> &x_shape, const PrimitivePtr &primitive) {
  auto x_rank = SizeToLong(x_shape.size());
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange("axis value", axis_temp, kIncludeLeft, {-x_rank, x_rank}, primitive->name());
  return axis_temp < 0 ? LongToSize(axis_temp + x_rank) : LongToSize(axis_temp);
}

bool IsDynamicOutputs(const std::vector<int64_t> &x_shape, const PrimitivePtr &primitive) {
  if (IsDynamicRank(x_shape)) {
    return true;
  }
  auto rank = x_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(rank), kGreaterEqual, 1, primitive->name());
  auto axis = GetUnstackAxis(x_shape, primitive);
  return x_shape[axis] == -1;
}

TypePtr UnstackInferType(const std::vector<AbstractBasePtr> &input_args, int64_t output_num) {
  auto type = input_args[kInputIndex0]->BuildType();
  std::vector<TypePtr> type_tuple;
  for (int64_t i = 0; i < output_num; ++i) {
    type_tuple.push_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}

BaseShapePtr UnstackInferShape(const std::vector<int64_t> &x_shape, size_t axis) {
  auto temp_shape = x_shape;
  (void)temp_shape.erase(temp_shape.begin() + SizeToLong(axis));
  std::vector<abstract::BaseShapePtr> shape_tuple;
  auto output_num = x_shape[axis];
  for (int64_t i = 0; i < output_num; ++i) {
    abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(temp_shape);
    shape_tuple.push_back(out_shape);
  }
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

AbstractBasePtr UnstackInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kUnstackInputsNum, prim_name);
  auto type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", type, common_valid_types_with_complex_and_bool,
                                                   prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicOutputs(x_shape, primitive)) {
    auto unstack_axis = GetUnstackAxis(x_shape, primitive);
    auto output_num = x_shape[unstack_axis];
    (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterThan, 0, prim_name);
    (void)primitive->AddAttr(kAttrNum, MakeValue(output_num));
    auto output_type = UnstackInferType(input_args, output_num);
    auto output_shape = UnstackInferShape(x_shape, unstack_axis);
    return abstract::MakeAbstract(output_shape, output_type);
  }
  // process dynamic num of outputs case
  BaseShapePtr shape;
  if (IsDynamicRank(x_shape)) {
    shape = std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  } else {  // the axis corresponding dim equals -1
    auto unstack_axis = GetUnstackAxis(x_shape, primitive);
    auto temp_shape = x_shape;
    (void)temp_shape.erase(temp_shape.begin() + SizeToLong(unstack_axis));
    shape = std::make_shared<abstract::Shape>(temp_shape);
  }
  auto dtype = type->cast<TensorTypePtr>()->element();
  auto output = std::make_shared<abstract::AbstractTensor>(dtype, shape);
  AbstractBasePtrList output_list = {output};
  auto output_abs_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
  output_abs_tuple->CheckAndConvertToDynamicLenSequence();
  return output_abs_tuple;
}
}  // namespace
class UnstackInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UnstackInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return UnstackInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UnstackInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Unstack, prim::kPrimUnstack, UnstackInfer, false);
}  // namespace ops
}  // namespace mindspore
