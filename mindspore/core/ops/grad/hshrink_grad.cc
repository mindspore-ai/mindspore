/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/hshrink_grad.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(HShrinkGrad, BaseOperator);
void HShrinkGrad::Init(const float &lambd) { this->set_lambd(lambd); }

void HShrinkGrad::set_lambd(const float &lambd) { (void)this->AddAttr(kLambd, api::MakeValue(lambd)); }

float HShrinkGrad::get_lambd() const {
  auto value_ptr = GetAttr(kLambd);
  return GetValue<float>(value_ptr);
}

namespace {
abstract::ShapePtr HShrinkGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  auto prim_name = primitive->name();
  auto gradients_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto features_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];

  CheckAndConvertUtils::Check("gradients_shape", gradients_shape, kEqual, features_shape, prim_name, TypeError);
  return std::make_shared<abstract::Shape>(gradients_shape);
}

TypePtr HShrinkGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)types.emplace("gradients", input_args[0]->BuildType());
  (void)types.emplace("features", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

AbstractBasePtr HShrinkGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = HShrinkGradInferType(primitive, input_args);
  auto infer_shape = HShrinkGradInferShape(primitive, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}
REGISTER_PRIMITIVE_EVAL_IMPL(HShrinkGrad, prim::kPrimHShrinkGrad, HShrinkGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
