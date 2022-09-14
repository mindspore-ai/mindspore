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

#include "ops/apply_momentum.h"

#include <utility>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void ApplyMomentum::Init(const bool use_nesterov, const bool use_locking, const float gradient_scale) {
  this->set_use_nesterov(use_nesterov);
  this->set_use_locking(use_locking);
  this->set_gradient_scale(gradient_scale);
}

void ApplyMomentum::set_use_nesterov(const bool use_nesterov) {
  (void)this->AddAttr(kUseNesterov, api::MakeValue(use_nesterov));
}

void ApplyMomentum::set_use_locking(const bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

void ApplyMomentum::set_gradient_scale(const float gradient_scale) {
  (void)this->AddAttr(kGradientScale, api::MakeValue(gradient_scale));
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
namespace {
abstract::ShapePtr ApplyMomentumInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 5;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Infer shape
  auto v_shape = input_args[kInputIndex0]->BuildShape();
  if (IsDynamicRank(CheckAndConvertUtils::ConvertShapePtrToShapeMap(v_shape)[kShape])) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (IsDynamicRank(a_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto l_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(l_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto g_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (IsDynamicRank(g_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  if (IsDynamicRank(m_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto shape_element = v_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}
TypePtr ApplyMomentumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 5;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Infer type
  auto v_tensor_type = input_args[kInputIndex0]->BuildType();
  auto a_tensor_type = input_args[kInputIndex1]->BuildType();
  auto l_type = input_args[kInputIndex2]->BuildType();
  auto g_type = input_args[kInputIndex3]->BuildType();
  auto m_type = input_args[kInputIndex4]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("v_type", v_tensor_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("a_type", a_tensor_type, valid_types, prim_name);
  std::map<std::string, TypePtr> args_l;
  (void)args_l.insert(std::make_pair("l_type", l_type));
  std::map<std::string, TypePtr> args_g;
  (void)args_g.insert(std::make_pair("g_type", g_type));
  std::map<std::string, TypePtr> args_m;
  (void)args_m.insert(std::make_pair("m_type", m_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_g, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_m, valid_types, prim_name);
  return v_tensor_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ApplyMomentum, BaseOperator);
AbstractBasePtr ApplyMomentumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = ApplyMomentumInferType(primitive, input_args);
  auto infer_shape = ApplyMomentumInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ApplyMomentum, prim::kPrimApplyMomentum, ApplyMomentumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
