/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <set>
#include <string>
#include "ops/mvlgamma.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MvlgammaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  auto first_input_shape_min = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMinShape];
  auto first_input_shape_max = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMaxShape];
  if (first_input_shape_min.empty() || first_input_shape_max.empty()) {
    return std::make_shared<abstract::Shape>(in_shape);
  }

  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr MvlgammaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, primitive->name());
}
}  // namespace

void Mvlgamma::Init(const int64_t p) { set_p(p); }

void Mvlgamma::set_p(const int64_t p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

int64_t Mvlgamma::get_p() const { return GetValue<int64_t>(GetAttr(kP)); }

AbstractBasePtr MvlgammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MvlgammaInferType(primitive, input_args);
  auto infer_shape = MvlgammaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Mvlgamma, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Mvlgamma, prim::kPrimMvlgamma, MvlgammaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
