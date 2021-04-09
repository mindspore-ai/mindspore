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

#include <set>

#include "ops/non_max_suppression.h"

namespace mindspore {
namespace ops {
void NonMaxSuppression::set_center_point_box(const int64_t center_point_box) {
  AddAttr(kCenterPointBox, MakeValue(center_point_box));
}
int64_t NonMaxSuppression::get_center_point_box() const {
  auto value_ptr = this->GetAttr(kCenterPointBox);
  return GetValue<int64_t>(value_ptr);
}
void NonMaxSuppression::Init(const int64_t center_point_box) { this->set_center_point_box(center_point_box); }

AbstractBasePtr NonMaxSuppressionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto non_max_suppression_prim = primitive->cast<PrimNonMaxSuppressionPtr>();
  MS_EXCEPTION_IF_NULL(non_max_suppression_prim);
  MS_LOG(INFO) << "NonMaxSuppression infer shape in runtime.";
  return std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeInt32), std::vector<int64_t>{});
}
REGISTER_PRIMITIVE_C(kNameNonMaxSuppression, NonMaxSuppression);
}  // namespace ops
}  // namespace mindspore
