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

#include "ops/atan.h"

namespace mindspore {
namespace ops {
AbstractBasePtr AtanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("Atan_infer", SizeToLong(input_args.size()), kEqual, 1, prim_name);

  // Infer Shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto infer_shape = std::make_shared<abstract::Shape>(x_shape);

  // Infer Type
  auto dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt32};
  auto element = CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", dtype, valid_types, prim_name);
  return std::make_shared<abstract::AbstractTensor>(element, infer_shape->shape());
}
REGISTER_PRIMITIVE_C(kNameAtan, Atan);
}  // namespace ops
}  // namespace mindspore
