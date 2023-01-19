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
constexpr auto kAttrNum = "num";
constexpr int64_t kUnstackInputsNum = 1;
}  // namespace
MIND_API_OPERATOR_IMPL(Unstack, BaseOperator);

void Unstack::Init(const int64_t axis) { this->set_axis(axis); }
void Unstack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t Unstack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

class UnstackInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kUnstackInputsNum, primitive->name());
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];

    auto output_num = GetValue<int64_t>(primitive->GetAttr(kAttrNum));
    auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));

    auto temp_shape = x_shape;
    if (!IsDynamicRank(x_shape)) {
      auto x_rank = SizeToLong(x_shape.size());
      auto axis = axis_temp < 0 ? LongToSize(axis_temp + x_rank) : LongToSize(axis_temp);
      if (axis >= x_shape.size()) {
        MS_LOG(EXCEPTION) << "Axis should be less than " << x_rank << ", but got " << axis;
      }
      (void)temp_shape.erase(temp_shape.begin() + axis);
      if (!IsDynamic(x_shape)) {
        auto output_num_from_shape = x_shape[axis];
        auto name = primitive->name();
        (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kEqual, output_num_from_shape, name);
      }
    }

    std::vector<abstract::BaseShapePtr> shape_tuple;
    for (int64_t i = 0; i < output_num; ++i) {
      abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(temp_shape);
      shape_tuple.push_back(out_shape);
    }
    return std::make_shared<abstract::TupleShape>(shape_tuple);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kUnstackInputsNum, primitive->name());
    auto name = primitive->name();
    auto type = input_args[kInputIndex0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", type, common_valid_types_with_complex_and_bool, name);

    int64_t output_num;
    auto num_value = primitive->GetAttr(kAttrNum);
    MS_EXCEPTION_IF_NULL(num_value);
    if (!num_value->isa<None>()) {
      output_num = GetValue<int64_t>(num_value);
      (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterEqual, 1, name);
    } else {
      // Num attr is None, try to infer output num from shape.
      auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
      if (IsDynamicRank(x_shape)) {
        MS_LOG(EXCEPTION) << "Unstack cannot auto infer output size in dynamic rank case.";
      }

      auto x_rank = SizeToLong(x_shape.size());
      (void)CheckAndConvertUtils::CheckInteger("x_rank", x_rank, kGreaterEqual, 1, name);

      auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
      CheckAndConvertUtils::CheckInRange("axis value", axis_temp, kIncludeLeft, {-x_rank, x_rank}, name);
      auto axis = axis_temp < 0 ? LongToSize(axis_temp + x_rank) : LongToSize(axis_temp);

      output_num = x_shape[axis];
      (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterEqual, 1, name);
      (void)primitive->AddAttr(kAttrNum, MakeValue(output_num));
    }

    std::vector<TypePtr> type_tuple;
    for (int64_t i = 0; i < output_num; ++i) {
      type_tuple.push_back(type);
    }
    return std::make_shared<Tuple>(type_tuple);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Unstack, prim::kPrimUnstack, UnstackInfer, false);
}  // namespace ops
}  // namespace mindspore
