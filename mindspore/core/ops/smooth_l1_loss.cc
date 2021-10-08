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

namespace mindspore {
namespace ops {
void SmoothL1Loss::Init(const float beta) { this->set_beta(beta); }
void SmoothL1Loss::set_beta(const float beta) { (void)this->AddAttr(kBeta, MakeValue(beta)); }

float SmoothL1Loss::get_beta() const {
  auto value_ptr = this->GetAttr(kBeta);
  return GetValue<int32_t>(value_ptr);
}

AbstractBasePtr SmoothL1LossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);

  // Infer shape
  auto prediction = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto target = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("prediction shape", prediction, kEqual, "target shape", target, prim_name, TypeError);

  // Infer type
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  args.emplace("scale", input_args[0]->BuildType());
  args.emplace("bias", input_args[1]->BuildType());
  auto prediction_type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);

  return std::make_shared<abstract::AbstractTensor>(prediction_type, prediction);
}
REGISTER_PRIMITIVE_C(kNameSmoothL1Loss, SmoothL1Loss);
}  // namespace ops
}  // namespace mindspore
