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

#include "ops/bce_with_logits_loss.h"

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BCEWithLogitsLossInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto logits_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto logits_shape = logits_shape_map[kShape];
  auto label_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto label_shape = label_shape_map[kShape];
  if (IsDynamicRank(logits_shape) || IsDynamicRank(label_shape)) {
    auto ds_shape = std::vector<int64_t>{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(ds_shape);
  }
  if (!ObscureShapeEqual(logits_shape, label_shape) && !(IsDynamicRank(logits_shape) || IsDynamicRank(label_shape))) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the two input 'logits' and 'label' shape are not equal.";
  }
  auto weight_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto weight_shape_shape = weight_shape_map[kShape];
  auto pos_weight_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape());
  auto pos_weight_shape = pos_weight_shape_map[kShape];

  auto value_ptr = primitive->GetAttr(kReduction);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto reduction_value = GetValue<std::string>(value_ptr);

  auto broadcast_weight_shape = CalBroadCastShape(logits_shape, weight_shape_shape, op_name, "logits", "weight");
  auto broadcast_pos_weight_shape = CalBroadCastShape(logits_shape, pos_weight_shape, op_name, "logits", "pos_weight");
  if (broadcast_weight_shape != broadcast_pos_weight_shape) {
    MS_EXCEPTION(ValueError)
      << "For '" << op_name
      << "', the two input 'weight' and 'pos_weight' shape can not broadcast to logits and label.";
  }
  // For BCEWithLogitsLoss, if reduction in ('mean', 'sum'), output will be a scalar.
  if (reduction_value != "none") {
    std::vector<int64_t> broadcast_shape;
    return std::make_shared<abstract::Shape>(broadcast_shape);
  }
  return std::make_shared<abstract::Shape>(broadcast_pos_weight_shape);
}

TypePtr BCEWithLogitsLossInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  const int64_t input_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  const std::vector<std::string> input_args_name = {"logits", "label", "weight", "pos_weight"};
  const std::set<TypePtr> data_type_check_list = {kFloat16, kFloat32};
  for (size_t index = kInputIndex1; index < input_args.size(); ++index) {
    auto input_item = input_args.at(index);
    MS_EXCEPTION_IF_NULL(input_item);
    TypePtr input_type = input_item->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid(input_args_name.at(index), input_type, data_type_check_list,
                                                     op_name);
  }
  auto logits_input_item = input_args.at(kInputIndex0);
  MS_EXCEPTION_IF_NULL(logits_input_item);
  TypePtr logits_input_type = logits_input_item->BuildType();
  auto logits_input_name = input_args_name.at(kInputIndex0);
  return CheckAndConvertUtils::CheckTensorTypeValid(logits_input_name, logits_input_type, data_type_check_list,
                                                    op_name);
}
}  // namespace

void BCEWithLogitsLoss::set_reduction(const std::string &reduction) {
  (void)CheckAndConvertUtils::CheckString(kReduction, reduction, {"none", "sum", "mean"}, this->name());
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}

std::string BCEWithLogitsLoss::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}

void BCEWithLogitsLoss::Init(const std::string &reduction) { this->set_reduction(reduction); }

MIND_API_OPERATOR_IMPL(BCEWithLogitsLoss, BaseOperator);
AbstractBasePtr BCEWithLogitsLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  TypePtr output_type = BCEWithLogitsLossInferType(primitive, input_args);
  abstract::ShapePtr output_shape = BCEWithLogitsLossInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(output_type, output_shape->shape());
}

// AG means auto generated
class MIND_API AGBCEWithLogitsLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BCEWithLogitsLossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BCEWithLogitsLossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BCEWithLogitsLossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BCEWithLogitsLoss, prim::kPrimBCEWithLogitsLoss, AGBCEWithLogitsLossInfer, false);
}  // namespace ops
}  // namespace mindspore
