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

#include <memory>
#include <string>
#include <vector>
#include "include/common/utils/utils.h"

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ops/sequence_ops.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "ops/array_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "ops/sequence_unstack.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
constexpr auto kAttrNum = "num";
constexpr int64_t kSequenceUnstackInputsNum = 1;
void SequenceUnstack::Init(const int64_t axis) { this->set_axis(axis); }
void SequenceUnstack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t SequenceUnstack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

size_t GetUnstackAxis(const std::vector<int64_t> &x_shape, const PrimitivePtr &primitive) {
  auto x_rank = SizeToLong(x_shape.size());
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange("axis value", axis_temp, kIncludeLeft, {-x_rank, x_rank}, primitive->name());
  return axis_temp < 0 ? LongToSize(axis_temp + x_rank) : LongToSize(axis_temp);
}

MIND_API_OPERATOR_IMPL(SequenceUnstack, BaseOperator);
AbstractBasePtr SequenceUnstackInferInner(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto type = input_args[kInputIndex0]->BuildType();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kSequenceUnstackInputsNum, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", type, common_valid_types_with_complex_and_bool, op_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicShape(x_shape)) {
    BaseShapePtr out_shape;
    if (IsDynamicRank(x_shape)) {
      out_shape = std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    } else {
      auto unstack_axis = GetUnstackAxis(x_shape, primitive);
      auto temp_shape = x_shape;
      (void)temp_shape.erase(temp_shape.begin() + SizeToLong(unstack_axis));
      out_shape = std::make_shared<abstract::Shape>(temp_shape);
    }
    auto dtype = type->cast<TensorTypePtr>();
    auto output = std::make_shared<abstract::AbstractTensor>(dtype->element(), out_shape);
    AbstractBasePtrList output_list = {output};
    auto output_abs_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
    output_abs_tuple->CheckAndConvertToDynamicLenSequence();
    return output_abs_tuple;
  } else {
    auto axis = GetUnstackAxis(x_shape, primitive);
    int64_t all_shp = x_shape[axis];
    (void)primitive->AddAttr(kAttrNum, MakeValue(all_shp));

    auto temp_shape = x_shape;
    auto temp_shape_axis = temp_shape[axis];
    (void)temp_shape.erase(temp_shape.begin() + axis);
    abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(temp_shape);

    auto dtype = type->cast<TensorTypePtr>();
    AbstractBasePtrList output_list;
    for (int64_t j = 0; j < temp_shape_axis; ++j) {
      output_list.push_back(std::make_shared<abstract::AbstractTensor>(dtype->element(), out_shape));
    }
    auto output_abs_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
    return output_abs_tuple;
  }
}

class SequenceUnstackInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceUnstackInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceUnstackInferInner(primitive, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceUnstackInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceUnstack, prim::kPrimSequenceUnstack, SequenceUnstackInfer, false);
}  // namespace ops
}  // namespace mindspore
