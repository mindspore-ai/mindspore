/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/unstack_with_num.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void UnstackWithNum::Init(const int64_t num, const int64_t axis) {
  this->set_axis(axis);
  this->set_num(num);
}
void UnstackWithNum::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t UnstackWithNum::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
void UnstackWithNum::set_num(const int64_t num) { (void)AddAttr("num", api::MakeValue(num)); }
int64_t UnstackWithNum::get_num() const { return GetValue<int64_t>(GetAttr("num")); }

MIND_API_OPERATOR_IMPL(UnstackWithNum, BaseOperator);

class UnstackWithNumInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];

    auto output_num = GetValue<int64_t>(primitive->GetAttr("num"));
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
    auto name = primitive->name();
    auto type = input_args[kInputIndex0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", type, common_valid_types_with_complex_and_bool, name);

    auto output_num = GetValue<int64_t>(primitive->GetAttr("num"));
    (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterEqual, 1, name);

    std::vector<TypePtr> type_tuple;
    for (int64_t i = 0; i < output_num; ++i) {
      type_tuple.push_back(type);
    }
    return std::make_shared<Tuple>(type_tuple);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UnstackWithNum, prim::kPrimUnstackWithNum, UnstackWithNumInfer, false);
}  // namespace ops
}  // namespace mindspore
