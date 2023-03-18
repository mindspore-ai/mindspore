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

#include "ops/sequence_concat.h"

#include <vector>
#include <string>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SequenceConcatInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto queue = abstract::CheckArg<abstract::AbstractSequence>(op_name, input_args, 0);
  // The value of dynamic_len_element_abs is kValueAny, do not need to Broaden.
  if (queue->dynamic_len()) {
    auto element_abs = queue->dynamic_len_element_abs();
    MS_EXCEPTION_IF_NULL(element_abs);
    auto ret_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element_abs->BuildShape())[kShape];
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  if (queue->elements().empty()) {
    MS_LOG(EXCEPTION) << "For " << op_name << " length should not be 0.";
  }
  const int64_t kOneNum = 1;
  auto elements = queue->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           op_name);
  (void)primitive->AddAttr("N", MakeValue(SizeToLong(elements.size())));
  (void)primitive->AddAttr("inputNums", MakeValue(SizeToLong(elements.size())));
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  if (IsDynamicRank(element0_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  auto element0_rank = element0_shape.size();
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange<int64_t>(
    "Concat axis", axis_temp, kIncludeBoth, {-SizeToLong(element0_rank), SizeToLong(element0_rank) - kOneNum}, op_name);
  auto axis = axis_temp < 0 ? LongToSize(axis_temp + SizeToLong(element0_rank)) : LongToSize(axis_temp);
  int64_t all_shp = element0_shape[axis];
  for (size_t i = 1; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    auto elementi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    if (IsDynamicRank(elementi_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    (void)CheckAndConvertUtils::CheckInteger(elementi + " shape rank", SizeToLong(elementi_shape.size()), kEqual,
                                             SizeToLong(element0_shape.size()), op_name);
    for (size_t j = 0; j < element0_rank; ++j) {
      if (j != axis && elementi_shape[j] != element0_shape[j] && elementi_shape[j] != -1 && element0_shape[j] != -1) {
        MS_EXCEPTION(ValueError)
          << "For '" << op_name << "', element" << i
          << " shape in input can not concat with element0. To perform concat in the axis 0 "
             "direction, except for the 0th axis, all other axes must have the same shape. But got "
          << "element" << i << "_shape[" << j << "]: " << elementi_shape[j] << ", element0_shape[" << j
          << "]: " << element0_shape[j] << ".";
      }
    }
    all_shp = all_shp == -1 || elementi_shape[axis] == -1 ? -1 : all_shp + elementi_shape[axis];
  }
  auto ret_shape = element0_shape;
  ret_shape[axis] = all_shp;
  return std::make_shared<abstract::Shape>(ret_shape);
}

AbstractBasePtr SequenceConcatInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto queue = abstract::CheckArg<abstract::AbstractSequence>(op_name, input_args, 0);

  // The value of dynamic_len_element_abs is kValueAny, do not need to Broaden.
  if (queue->dynamic_len()) {
    auto element_abs = queue->dynamic_len_element_abs();
    MS_EXCEPTION_IF_NULL(element_abs);
    return element_abs->Clone();
  }

  if (queue->elements().empty()) {
    MS_LOG(EXCEPTION) << "Sequence length should not be 0.";
  }
  return queue->elements()[0];
}
}  // namespace

void SequenceConcat::Init(const int64_t axis) { this->set_axis(axis); }
int64_t SequenceConcat::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
void SequenceConcat::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

MIND_API_OPERATOR_IMPL(SequenceConcat, BaseOperator);
AbstractBasePtr SequenceConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = SequenceConcatInferType(primitive, input_args)->BuildType();
  auto infer_shape = SequenceConcatInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSequenceConcatInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceConcatInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceConcatInferType(primitive, input_args)->BuildType();
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceConcatInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceConcat, prim::kPrimSequenceConcat, AGSequenceConcatInfer, false);
}  // namespace ops
}  // namespace mindspore
