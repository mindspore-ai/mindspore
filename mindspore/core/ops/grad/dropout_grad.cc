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

#include "ops/grad/dropout_grad.h"
#include <string>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void DropoutGrad::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void DropoutGrad::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, MakeValue(keep_prob));
}

float DropoutGrad::get_keep_prob() const {
  auto value_ptr = GetAttr(kKeepProb);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr DropoutGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);

  const int64_t dy_index = 0;
  const int64_t mask_index = 1;
  auto dy_type = input_args[dy_index]->BuildType();
  auto mask_type = input_args[mask_index]->BuildType();

  (void)CheckAndConvertUtils::CheckTensorTypeValid("mask", mask_type, {kTensorType}, op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto out_type = CheckAndConvertUtils::CheckTensorTypeValid("x", dy_type, valid_types, op_name);
  auto shape = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, dy_index);
  return abstract::MakeAbstract(shape, out_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(DropoutGrad, prim::kPrimDropoutGrad, DropoutGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
