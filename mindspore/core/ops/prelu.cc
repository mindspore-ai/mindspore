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
#include "ops/prelu.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x = input_args[0]->BuildShape();
  auto w = input_args[1]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", x, prim_name);
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShape("w_shape", w, prim_name);

  CheckAndConvertUtils::CheckInteger("x rank", x_shape.size(), kNotEqual, 1, prim_name);
  CheckAndConvertUtils::CheckInteger("weight rank", w_shape.size(), kEqual, 1, prim_name);
  if (w_shape[0] != x_shape[1] && w_shape[0] != 1) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", channel of input_x and weight must be matched, "
                      << "while channel of input_x is " << x_shape[1] << ", weight_shape[0] is " << w_shape[0];
  }

  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 2, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim->name());
  CheckAndConvertUtils::CheckTensorTypeValid("weight", input_args[1]->BuildType(), valid_types, prim->name());
  auto tensor_type = input_args[0]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto input_x_type = tensor_type->element();
  return input_x_type;
}
}  // namespace
AbstractBasePtr PReLUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNamePReLU, PReLU);
}  // namespace ops
}  // namespace mindspore
