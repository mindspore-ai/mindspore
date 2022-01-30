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

#include "ops/adam.h"

#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
TuplePtr AdamInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto v_type = input_args[kInputIndex2]->BuildType();
  auto beta1_power_type = input_args[kInputIndex3]->BuildType();
  auto beta2_power_type = input_args[kInputIndex4]->BuildType();
  auto lr_type = input_args[kInputIndex5]->BuildType();
  auto beta1_type = input_args[kInputIndex6]->BuildType();
  auto beta2_type = input_args[kInputIndex7]->BuildType();
  auto epsilon_type = input_args[kInputIndex8]->BuildType();
  auto grad_type = input_args[kInputIndex9]->BuildType();
  std::map<std::string, TypePtr> type_dict;
  type_dict.emplace("var", var_type);
  type_dict.emplace("m", m_type);
  type_dict.emplace("v", v_type);
  type_dict.emplace("grad", grad_type);
  std::set<TypePtr> num_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(type_dict, num_type, prim_name);
  std::map<std::string, TypePtr> type_dict1;
  type_dict1.emplace("beta1_power", beta1_power_type);
  type_dict1.emplace("beta2_power", beta2_power_type);
  type_dict1.emplace("lr", lr_type);
  type_dict1.emplace("beta1", beta1_type);
  type_dict1.emplace("beta2", beta2_type);
  type_dict1.emplace("epsilon", epsilon_type);
  std::set<TypePtr> float_set = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(type_dict1, float_set, prim_name, true);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type});
}
abstract::TupleShapePtr AdamInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto m_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto v_shape_ptr = input_args[kInputIndex2]->BuildShape();
  auto grad_shape_ptr = input_args[kInputIndex9]->BuildShape();
  if (var_shape_ptr->IsDynamic() || m_shape_ptr->IsDynamic() || v_shape_ptr->IsDynamic() ||
      grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{var_shape_ptr, m_shape_ptr, v_shape_ptr});
  }
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, m_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, v_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, grad_shape, prim_name);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape_ptr, m_shape_ptr, v_shape_ptr});
}
}  // namespace
void Adam::Init(const bool use_locking, const bool use_nesterov) {
  this->set_use_locking(use_locking);
  this->set_use_nesterov(use_nesterov);
}

void Adam::set_use_locking(const bool use_locking) { (void)this->AddAttr(kUseLocking, MakeValue(use_locking)); }

void Adam::set_use_nesterov(const bool use_nesterov) { (void)this->AddAttr(kUseNesterov, MakeValue(use_nesterov)); }

bool Adam::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

bool Adam::get_use_nesterov() const {
  auto value_ptr = GetAttr(kUseNesterov);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr AdamInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNum = 10;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = AdamInferType(primitive, input_args);
  auto infer_shape = AdamInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Adam, prim::kPrimAdam, AdamInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
