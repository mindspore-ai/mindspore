/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <string>
#include <vector>

#include "ops/smooth_l1_loss.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(SmoothL1Loss, PrimitiveC, BaseOperator);
void SmoothL1Loss::Init(const float beta) { this->set_beta(beta); }
void SmoothL1Loss::set_beta(const float beta) { (void)this->AddAttr(kBeta, api::MakeValue(beta)); }

float SmoothL1Loss::get_beta() const {
  auto value_ptr = this->GetAttr(kBeta);
  return GetValue<float>(value_ptr);
}

namespace {
abstract::ShapePtr SmoothL1LossInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto prediction = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto target = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  abstract::CheckShapeSame(prim_name, prediction, target);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SmoothL1LossInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  // Infer type
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("scale", input_args[kInputIndex0]->BuildType());
  (void)args.emplace("bias", input_args[kInputIndex1]->BuildType());
  auto prediction_type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim->name());
  return prediction_type;
}
}  // namespace

AbstractBasePtr SmoothL1LossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = SmoothL1LossInferType(primitive, input_args);
  auto infer_shape = SmoothL1LossInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SmoothL1Loss, prim::kPrimSmoothL1Loss, SmoothL1LossInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
