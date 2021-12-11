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

#include <vector>
#include <set>
#include <map>
#include <string>
#include "ops/grad/smooth_l1_loss_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void SmoothL1LossGrad::Init(const float beta) { this->set_beta(beta); }

void SmoothL1LossGrad::set_beta(const float beta) { (void)this->AddAttr(kBeta, MakeValue(beta)); }

float SmoothL1LossGrad::get_beta() const {
  auto value_ptr = this->GetAttr(kBeta);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int32_t>(value_ptr);
}

namespace {
abstract::ShapePtr SmoothL1LossGradInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto prediction = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto target = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  auto dloss = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  abstract::CheckShapeSame(prim_name, prediction, target);
  abstract::CheckShapeSame(prim_name, prediction, dloss);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SmoothL1LossGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  // Infer type
  const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("prediction", input_args[kInputIndex0]->BuildType());
  (void)args.emplace("target", input_args[kInputIndex1]->BuildType());
  (void)args.emplace("dloss", input_args[kInputIndex2]->BuildType());
  auto dloss_type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim->name());
  return dloss_type;
}
}  // namespace

AbstractBasePtr SmoothL1LossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = SmoothL1LossGradInferType(primitive, input_args);
  auto infer_shape = SmoothL1LossGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SmoothL1LossGrad, prim::kPrimSmoothL1LossGrad, SmoothL1LossGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
