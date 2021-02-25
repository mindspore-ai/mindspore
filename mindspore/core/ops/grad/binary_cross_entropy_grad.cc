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
#include <map>
#include <string>

#include "ops/grad/binary_cross_entropy_grad.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BinaryCrossEntroyGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto binary_cross_entropy_grad_prim = primitive->cast<PrimBinaryCrossEntropyGradPtr>();
  MS_EXCEPTION_IF_NULL(binary_cross_entropy_grad_prim);
  auto prim_name = binary_cross_entropy_grad_prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShape("y_shape", input_args[1]->BuildShape(), prim_name);
  auto weight_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("weight_shape", input_args[2]->BuildShape(), prim_name);
  CheckAndConvertUtils::Check("x shape", x_shape, kEqual, "y shape", y_shape, prim_name);
  if (weight_shape.size() < 1) {
    CheckAndConvertUtils::Check("y shape", y_shape, kEqual, "weight shape", weight_shape, prim_name);
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr BinaryCrossEntroyGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("x_shape", input_args[0]->BuildType());
  types.emplace("y_shape", input_args[1]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  if (input_args[3]->BuildType() != nullptr) {
    types.emplace("x_shape", input_args[0]->BuildType());
    types.emplace("weight_shape", input_args[2]->BuildType());
    infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  }
  return TypeIdToType(infer_type);
}
}  // namespace
void BinaryCrossEntropyGrad::Init(const Reduction &reduction) { set_reduction(reduction); }

void BinaryCrossEntropyGrad::set_reduction(const Reduction &reduction) {
  int64_t swi = reduction;
  this->AddAttr(kReduction, MakeValue(swi));
}
Reduction BinaryCrossEntropyGrad::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return Reduction(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr BinaryCrossEntropyGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(BinaryCrossEntroyGradInferType(primitive, input_args),
                                                    BinaryCrossEntroyGradInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropyGrad, BinaryCrossEntropyGrad);
}  // namespace ops
}  // namespace mindspore
