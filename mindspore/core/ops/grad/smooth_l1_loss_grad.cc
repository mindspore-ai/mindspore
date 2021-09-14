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

AbstractBasePtr SmoothL1LossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("smooth_l1_loss_grad_infer", SizeToLong(input_args.size()), kEqual,
                                           input_num, prim_name);

  // Infer shape
  auto prediction = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto target = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto dloss = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("prediction shape", prediction, kEqual, "target shape", target, prim_name, TypeError);
  CheckAndConvertUtils::Check("prediction shape", prediction, kEqual, "dloss", dloss, prim_name, TypeError);

  // Infer type
  const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("prediction", input_args[kInputIndex0]->BuildType());
  (void)args.emplace("target", input_args[kInputIndex1]->BuildType());
  (void)args.emplace("dloss", input_args[kInputIndex2]->BuildType());
  auto dloss_type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  return std::make_shared<abstract::AbstractTensor>(dloss_type, prediction);
}
REGISTER_PRIMITIVE_C(kNameSmoothL1LossGrad, SmoothL1LossGrad);
}  // namespace ops
}  // namespace mindspore
