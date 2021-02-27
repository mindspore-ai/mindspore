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

#include <map>
#include <string>
#include <memory>
#include <vector>
#include "ops/fusion/pow_fusion.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void PowFusion::Init(const float &scale, const float &shift) {
  this->set_scale(scale);
  this->set_shift(shift);
}

void PowFusion::set_scale(const float &scale) { this->AddAttr(kScale, MakeValue(scale)); }
void PowFusion::set_shift(const float &shift) { this->AddAttr(kShift, MakeValue(shift)); }

float PowFusion::get_scale() const { return GetValue<float>(GetAttr(kScale)); }
float PowFusion::get_shift() const { return GetValue<float>(GetAttr(kShift)); }

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto pow_prim = primitive->cast<PrimPowPtr>();
  MS_EXCEPTION_IF_NULL(pow_prim);
  auto op_name = pow_prim->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("y", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace

AbstractBasePtr PowFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNamePowFusion, PowFusion);
}  // namespace ops
}  // namespace mindspore
