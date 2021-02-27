/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

void SmoothL1LossGrad::set_beta(const float beta) { this->AddAttr(kBeta, MakeValue(beta)); }

float SmoothL1LossGrad::get_beta() const {
  auto value_ptr = this->GetAttr(kBeta);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr SmoothL1LossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto smooth_prim = primitive->cast<PrimSmoothL1LossGradPtr>();
  MS_EXCEPTION_IF_NULL(smooth_prim);
  auto prim_name = smooth_prim->name();
  CheckAndConvertUtils::CheckInteger("smooth_l1_loss_grad_infer", input_args.size(), kEqual, 3, prim_name);

  // Infer shape
  auto prediction = CheckAndConvertUtils::ConvertShapePtrToShape("prediction", input_args[0]->BuildShape(), prim_name);
  auto target = CheckAndConvertUtils::ConvertShapePtrToShape("target", input_args[1]->BuildShape(), prim_name);
  auto dloss = CheckAndConvertUtils::ConvertShapePtrToShape("dloss", input_args[2]->BuildShape(), prim_name);
  CheckAndConvertUtils::Check("prediction shape", prediction, kEqual, "target shape", target, prim_name, TypeError);
  CheckAndConvertUtils::Check("prediction shape", prediction, kEqual, "dloss", dloss, prim_name, TypeError);

  // Infer type
  const std::set<TypeId> valid_types = {
    kNumberTypeBool,    kNumberTypeInt,     kNumberTypeInt8,    kNumberTypeInt16,
    kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt,    kNumberTypeUInt8,
    kNumberTypeUInt16,  kNumberTypeUInt32,  kNumberTypeUInt64,  kNumberTypeFloat,
    kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex64};
  std::map<std::string, TypePtr> args;
  args.emplace("prediction", input_args[0]->BuildType());
  args.emplace("target", input_args[1]->BuildType());
  args.emplace("dloss", input_args[2]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  auto dloss_type = input_args[2]->BuildType()->cast<TensorTypePtr>()->element();

  return std::make_shared<abstract::AbstractTensor>(dloss_type, prediction);
}
REGISTER_PRIMITIVE_C(kNameSmoothL1LossGrad, SmoothL1LossGrad);
}  // namespace ops
}  // namespace mindspore
