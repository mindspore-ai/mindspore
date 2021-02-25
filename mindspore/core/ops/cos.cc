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
#include "ops/cos.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto cos_prim = primitive->cast<PrimCos>();
  MS_EXCEPTION_IF_NULL(cos_prim);
  auto prim_name = cos_prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](AbstractBasePtr a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace

AbstractBasePtr CosInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameCos, Cos);
}  // namespace ops
}  // namespace mindspore
