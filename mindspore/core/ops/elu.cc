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

#include <string>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include "ops/elu.h"

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto elu_prim = primitive->cast<PrimElu>();
  MS_EXCEPTION_IF_NULL(elu_prim);
  auto op_name = elu_prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->GetShapeTrack(), op_name);
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  types.emplace("x", input_args[0]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
}
}  // namespace
void Elu::Init(const float alpha) { this->set_alpha(alpha); }

void Elu::set_alpha(const float alpha) {
  AddAttr(kAlpha, MakeValue(CheckAndConvertUtils::CheckValue<float>(kAlpha, alpha, kEqual, 1.0, name())));
}

float Elu::get_alpha() const {
  auto value_ptr = this->GetAttr(kAlpha);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr EluInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameElu, Elu);
}  // namespace ops
}  // namespace mindspore
