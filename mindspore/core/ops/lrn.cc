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

#include "ops/lrn.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void LRN::set_depth_radius(const int64_t depth_radius) {
  CheckAndConvertUtils::CheckInteger(kDepthRadius, depth_radius, kGreaterEqual, 0, this->name());
  this->AddAttr(kDepthRadius, MakeValue(depth_radius));
}

int64_t LRN::get_depth_radius() const {
  auto value_ptr = GetAttr(kDepthRadius);
  return GetValue<int64_t>(value_ptr);
}

void LRN::set_bias(const float bias) { this->AddAttr(kBias, MakeValue(bias)); }

float LRN::get_bias() const {
  auto value_ptr = GetAttr(kBias);
  return GetValue<float>(value_ptr);
}

void LRN::set_alpha(const float alpha) { this->AddAttr(kAlpha, MakeValue(alpha)); }

float LRN::get_alpha() const {
  auto value_ptr = GetAttr(kAlpha);
  return GetValue<float>(value_ptr);
}

void LRN::set_beta(const float beta) { this->AddAttr(kBeta, MakeValue(beta)); }

float LRN::get_beta() const {
  auto value_ptr = GetAttr(kBeta);
  return GetValue<float>(value_ptr);
}
void LRN::set_norm_region(const std::string &norm_region) {
  CheckAndConvertUtils::CheckString(kNormRegion, norm_region, {"ACROSS_CHANNELS"}, this->name());
  this->AddAttr(kNormRegion, MakeValue(norm_region));
}

std::string LRN::get_norm_region() const {
  auto value_ptr = GetAttr(kNormRegion);
  return GetValue<std::string>(value_ptr);
}
void LRN::Init(const int64_t depth_radius, const float bias, const float alpha, const float beta,
               const std::string &norm_region) {
  this->set_depth_radius(depth_radius);
  this->set_bias(bias);
  this->set_alpha(alpha);
  this->set_beta(beta);
  this->set_norm_region(norm_region);
}

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto lrn_prim = primitive->cast<PrimLrn>();
  MS_EXCEPTION_IF_NULL(lrn_prim);
  auto prim_name = lrn_prim->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("input shape", in_shape.size(), kEqual, 4, prim_name);
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  if (std::any_of(input_args.begin(), input_args.end(), [](AbstractBasePtr a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace

AbstractBasePtr LrnInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameLRN, LRN);
}  // namespace ops
}  // namespace mindspore
