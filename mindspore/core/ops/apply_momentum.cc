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
#include <memory>
#include <set>
#include <map>
#include <string>
#include "ops/apply_momentum.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void ApplyMomentum::Init(const bool use_nesterov, const bool use_locking, const float gradient_scale) {
  this->set_use_nesterov(use_nesterov);
  this->set_use_locking(use_locking);
  this->set_gradient_scale(gradient_scale);
}

void ApplyMomentum::set_use_nesterov(const bool use_nesterov) {
  (void)this->AddAttr(kUseNesterov, MakeValue(use_nesterov));
}

void ApplyMomentum::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, MakeValue(use_locking));
}

void ApplyMomentum::set_gradient_scale(const float gradient_scale) {
  (void)this->AddAttr(kGradientScale, MakeValue(gradient_scale));
}

bool ApplyMomentum::get_use_nesterov() const {
  auto value_ptr = GetAttr(kUseNesterov);
  return GetValue<bool>(value_ptr);
}

bool ApplyMomentum::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

float ApplyMomentum::get_gradient_scale() const {
  auto value_ptr = GetAttr(kGradientScale);
  return GetValue<float>(value_ptr);
}
AbstractBasePtr ApplyMomentumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Infer shape
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];

  // Infer type
  auto v_tensor_type = input_args[kInputIndex0]->BuildType();
  auto a_tensor_type = input_args[kInputIndex1]->BuildType();
  auto l_type = input_args[kInputIndex2]->BuildType();
  auto g_type = input_args[kInputIndex3]->BuildType();
  auto m_type = input_args[kInputIndex4]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("v_type", v_tensor_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("a_type", a_tensor_type, valid_types, prim_name);
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("l_type", l_type));
  (void)args.insert(std::make_pair("g_type", g_type));
  (void)args.insert(std::make_pair("m_type", m_type));
  CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, valid_types, prim_name);
  auto g_type_tensor = g_type->cast<TensorTypePtr>();
  auto element = g_type_tensor->element();
  return std::make_shared<abstract::AbstractTensor>(element, v_shape);
}
REGISTER_PRIMITIVE_C(kNameApplyMomentum, ApplyMomentum);
}  // namespace ops
}  // namespace mindspore
