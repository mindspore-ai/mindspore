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

#include <memory>
#include "ops/grad/batch_norm_grad_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kBatchNormGradGradInputsNum = 8;
}  // namespace

void BatchNormGradGrad::Init(bool is_training, float epsilon, const std::string &format) {
  this->set_is_training(is_training);
  this->set_epsilon(epsilon);
  this->set_format(format);
}

void BatchNormGradGrad::set_epsilon(float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float BatchNormGradGrad::get_epsilon() const {
  auto epsilon = this->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(epsilon);
  return GetValue<float>(epsilon);
}

void BatchNormGradGrad::set_is_training(bool is_training) {
  (void)this->AddAttr(kIsTraining, api::MakeValue(is_training));
}

bool BatchNormGradGrad::get_is_training() const {
  auto is_training = this->GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(is_training);
  return GetValue<bool>(is_training);
}

void BatchNormGradGrad::set_format(const std::string &format) { (void)this->AddAttr(kFormat, api::MakeValue(format)); }

std::string BatchNormGradGrad::get_format() const {
  auto format = this->GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(format);
  return GetValue<std::string>(format);
}

abstract::TupleShapePtr BatchNormGradGradInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kBatchNormGradGradInputsNum, prim_name);
  BaseShapePtr x_shape = input_args[kInputIndex0]->BuildShape();
  BaseShapePtr dy_shape = input_args[kInputIndex1]->BuildShape();
  BaseShapePtr scale_shape = input_args[kInputIndex2]->BuildShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, dy_shape, scale_shape});
}

TuplePtr BatchNormGradGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kBatchNormGradGradInputsNum, prim_name);
  TypePtr x_type = input_args[kInputIndex0]->BuildType();
  TypePtr dy_type = input_args[kInputIndex1]->BuildType();
  TypePtr scale_type = input_args[kInputIndex2]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, dy_type, scale_type});
}

AbstractBasePtr BatchNormGradGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BatchNormGradGradInferShape(primitive, input_args),
                                BatchNormGradGradInferType(primitive, input_args));
}

MIND_API_OPERATOR_IMPL(BatchNormGradGrad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(BatchNormGradGrad, prim::kPrimBatchNormGradGrad, BatchNormGradGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
