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
#include <memory>

#include "ops/gelu.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GeluInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto gelu_prim = primitive->cast<PrimGeluPtr>();
  MS_EXCEPTION_IF_NULL(gelu_prim);
  auto prim_name = gelu_prim->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_x", input_args[0]->BuildShape(), prim_name);
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr GeluInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("input_x", input_args[0]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace
AbstractBasePtr GeluInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(GeluInferType(primitive, input_args),
                                                    GeluInferShape(primitive, input_args)->shape());
}

REGISTER_PRIMITIVE_EVAL_IMPL(Gelu, prim::kPrimGelu, GeluInfer);
REGISTER_PRIMITIVE_C(kNameGelu, Gelu);
}  // namespace ops
}  // namespace mindspore
