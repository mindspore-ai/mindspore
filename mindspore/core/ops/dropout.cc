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
#include <vector>
#include <memory>

#include "ops/dropout.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void Dropout::Init(const float keep_prob) { this->set_keep_prob(keep_prob); }

void Dropout::set_keep_prob(const float keep_prob) {
  CheckAndConvertUtils::CheckInRange<float>(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kKeepProb, MakeValue(keep_prob));
}

float Dropout::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr DropoutInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("dropout_infer", SizeToLong(input_args.size()), kEqual, 1, prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x_shape", SizeToLong(x_shape.size()), kGreaterEqual, 1, prim_name);
  std::vector<int64_t> out_shape;
  (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end());
  (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end());
  auto infer_shape = std::make_shared<abstract::Shape>(out_shape);

  // Infer type
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto infer_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", input_args[0]->BuildType(), valid_types, prim_name);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}
REGISTER_PRIMITIVE_C(kNameDropout, Dropout);
}  // namespace ops
}  // namespace mindspore
