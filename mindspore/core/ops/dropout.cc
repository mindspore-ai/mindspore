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
  this->AddAttr(kKeepProb, MakeValue(keep_prob));
}

float Dropout::get_keep_prob() const {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr DropoutInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto dropout_prim = primitive->cast<PrimDropoutPtr>();
  MS_EXCEPTION_IF_NULL(dropout_prim);
  auto prim_name = dropout_prim->name();
  CheckAndConvertUtils::CheckInteger("dropout_infer", input_args.size(), kEqual, 1, prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("x_shape", x_shape.size(), kGreaterEqual, 1, prim_name);
  std::vector<int64_t> out_shape;
  out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end());
  out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end());
  auto infer_shape = std::make_shared<abstract::Shape>(out_shape);

  // Infer type
  auto dtype = input_args[0]->BuildType();
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", dtype, valid_types, prim_name);
  auto tensor_type = dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element);
  auto infer_type = std::make_shared<TensorType>(TypeIdToType(element->type_id()));

  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}
REGISTER_PRIMITIVE_C(kNameDropout, Dropout);
}  // namespace ops
}  // namespace mindspore
