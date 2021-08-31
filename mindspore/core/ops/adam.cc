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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::AbstractBasePtr AdamInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 10;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, prim_name);

  // infer shape
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShapeTrack())[kShape];
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShapeTrack())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->GetShapeTrack())[kShape];
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, "m_shape", m_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, "v_shape", v_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, "grad_shape", grad_shape, prim_name);

  // infer type
  auto var_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto v_type = input_args[kInputIndex2]->BuildType();
  auto grad_type = input_args[kInputIndex9]->BuildType();
  auto infer_var_type = CheckAndConvertUtils::CheckTensorTypeValid("var_type", var_type, common_valid_types, prim_name);
  auto infer_m_type = CheckAndConvertUtils::CheckTensorTypeValid("m_type", m_type, common_valid_types, prim_name);
  auto infer_v_type = CheckAndConvertUtils::CheckTensorTypeValid("v_type", v_type, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad_type", grad_type, common_valid_types, prim_name);
  auto output0 = std::make_shared<abstract::AbstractTensor>(infer_var_type, var_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(infer_m_type, m_shape);
  auto output2 = std::make_shared<abstract::AbstractTensor>(infer_v_type, v_shape);
  AbstractBasePtrList output = {output0, output1, output2};
  return std::make_shared<abstract::AbstractTuple>(output);
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
  return std::make_shared<abstract::AbstractTensor>(AdamInfer(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameAdam, Adam);
}  // namespace ops
}  // namespace mindspore
