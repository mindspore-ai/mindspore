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

#include "ops/fused_ada_factor.h"
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kParamIndex = 7;
constexpr size_t kFusedAdaFactorInputsNum = 12;
auto constexpr kEnableScaleParameter = "enable_scale_parameter";
auto constexpr kEnableFirstMoment = "enable_first_moment";
auto constexpr kEnableWeightDecay = "enable_weight_decay";
abstract::TupleShapePtr FusedAdaFactorInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto param_shape_r = input_args[kParamIndex]->Broaden()->BuildShape();
  auto outputs = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>({param_shape_r}));
  return outputs;
}

TypePtr FusedAdaFactorInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto type = input_args[kParamIndex]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{type});
}
}  // namespace

void FusedAdaFactor::set_enable_scale_parameter(bool flag) {
  (void)this->AddAttr(kEnableScaleParameter, api::MakeValue(flag));
}

bool FusedAdaFactor::get_enable_scale_parameter() const {
  auto value_ptr = GetAttr(kEnableScaleParameter);
  return GetValue<bool>(value_ptr);
}

void FusedAdaFactor::set_enable_first_moment(bool flag) {
  (void)this->AddAttr(kEnableFirstMoment, api::MakeValue(flag));
}

bool FusedAdaFactor::get_enable_first_moment() const {
  auto value_ptr = GetAttr(kEnableFirstMoment);
  return GetValue<bool>(value_ptr);
}

void FusedAdaFactor::set_enable_weight_decay(bool flag) {
  (void)this->AddAttr(kEnableWeightDecay, api::MakeValue(flag));
}

bool FusedAdaFactor::get_enable_weight_decay() const {
  auto value_ptr = GetAttr(kEnableWeightDecay);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(FusedAdaFactor, BaseOperator);
MIND_API_OPERATOR_IMPL(FusedAdaFactorWithGlobalNorm, FusedAdaFactor);
AbstractBasePtr FusedAdaFactorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           SizeToLong(kFusedAdaFactorInputsNum), op_name);
  auto types = FusedAdaFactorInferType(primitive, input_args);
  auto shapes = FusedAdaFactorInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(FusedAdaFactor, prim::kPrimFusedAdaFactor, FusedAdaFactorInfer, nullptr, true)
REGISTER_PRIMITIVE_EVAL_IMPL(FusedAdaFactorWithGlobalNorm, prim::kPrimFusedAdaFactorWithGlobalNorm, FusedAdaFactorInfer,
                             nullptr, true)
}  // namespace ops
}  // namespace mindspore
