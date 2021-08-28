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

#include "ops/l2_normalize.h"

namespace mindspore {
namespace ops {
void L2Normalize::Init(const std::vector<int64_t> &axis, const float epsilon) {
  this->set_axis(axis);
  this->set_epsilon(epsilon);
}

void L2Normalize::set_axis(const std::vector<int64_t> &axis) { (void)AddAttr(kAxis, MakeValue(axis)); }

void L2Normalize::set_epsilon(const float epsilon) { (void)AddAttr(kEpsilon, MakeValue(epsilon)); }

std::vector<int64_t> L2Normalize::get_axis() const { return GetValue<std::vector<int64_t>>(GetAttr(kAxis)); }

float L2Normalize::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr L2NormalizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", int64_t(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  auto axiss = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAxis));
  for (auto &axis : axiss) {
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-x_rank, x_rank}, prim_name);
  }
  return input_args[0]->Broaden();
}
REGISTER_PRIMITIVE_C(kNameL2Normalize, L2Normalize);
}  // namespace ops
}  // namespace mindspore
