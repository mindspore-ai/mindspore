/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <string>
#include "ops/soft_margin_loss.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSoftMarginLossInputSize = 2;
abstract::ShapePtr SoftMarginLossInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           kSoftMarginLossInputSize, op_name);
  auto predict = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto label = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("logits shape", predict, kEqual, label, op_name, ValueError);
  auto out_shape = predict;
  int64_t reduction;
  CheckAndConvertUtils::GetReductionEnumValue(primitive->GetAttr(kReduction), &reduction);
  if (reduction == REDUCTION_SUM || reduction == MEAN) {
    out_shape.resize(0);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr SoftMarginLossInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           kSoftMarginLossInputSize, op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("logits", input_args[0]->BuildType());
  (void)types.emplace("labels", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return input_args[0]->BuildType();
}
}  // namespace

void SoftMarginLoss::set_reduction(const std::string &reduction) {
  (void)CheckAndConvertUtils::CheckString(kReduction, reduction, {"none", "sum", "mean"}, this->name());
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}

std::string SoftMarginLoss::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}

void SoftMarginLoss::Init(const std::string &reduction) { this->set_reduction(reduction); }

MIND_API_OPERATOR_IMPL(SoftMarginLoss, BaseOperator);
AbstractBasePtr SoftMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SoftMarginLossInferShape(primitive, input_args),
                                SoftMarginLossInferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(SoftMarginLoss, prim::kPrimSoftMarginLoss, SoftMarginLossInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
