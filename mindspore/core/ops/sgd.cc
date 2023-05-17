/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/sgd.h"

#include "utils/check_convert_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace sgd {
// "parameters", "gradient", "learning_rate", "accum", "momentum", "stat"
constexpr size_t kParametersIndex = 0;
constexpr size_t kGradientIndex = 1;
constexpr size_t kLearningRateIndex = 2;
constexpr size_t kAccumIndex = 3;
constexpr size_t kMomentumIndex = 4;
constexpr size_t kStatIndex = 5;
constexpr size_t kSGDInputNum = 6;

abstract::BaseShapePtr SgdInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto parameters_shape_r = input_args[kParametersIndex]->Broaden()->BuildShape();
  for (auto &input : input_args) {
    if (input->BuildShape()->IsDynamic()) {
      return parameters_shape_r;
    }
  }
  auto parameters_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kParametersIndex]->BuildShape())[kShape];
  auto gradient_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGradientIndex]->BuildShape())[kShape];
  auto stat_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kStatIndex]->BuildShape())[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kAccumIndex]->BuildShape())[kShape];

  auto learning_rate_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kLearningRateIndex]->BuildShape())[kShape];
  auto momentum_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kMomentumIndex]->BuildShape())[kShape];
  auto is_scalar_shape = [](const std::vector<int64_t> &shape) {
    return shape.empty() || (shape.size() == 1 && shape[0] == 1);
  };
  if (!is_scalar_shape(learning_rate_shape)) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name
                             << "], the [learning rate] should be a scalar. but got shape [" << learning_rate_shape
                             << "]";
  }
  if (!is_scalar_shape(momentum_shape)) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the [momentum] should be a scalar. but got shape ["
                             << momentum_shape << "]";
  }
  return parameters_shape_r;
}

TypePtr SdgInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  // "parameters", "gradient", "learning_rate", "accum", "momentum", "stat"
  auto prim_name = prim->name();

  (void)CheckAndConvertUtils::CheckTensorTypeValid("parameters", input_args[kParametersIndex]->BuildType(),
                                                   {kFloat32, kFloat16}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("gradient", input_args[kGradientIndex]->BuildType(),
                                                   {kFloat32, kFloat16}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("momentum", input_args[kMomentumIndex]->BuildType(),
                                                   {kFloat32, kFloat16}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("stat", input_args[kStatIndex]->BuildType(), {kFloat32, kFloat16},
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("learning_rate", input_args[kLearningRateIndex]->BuildType(),
                                                   {kFloat32, kFloat16}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("accum", input_args[kAccumIndex]->BuildType(), {kFloat32, kFloat16},
                                                   prim_name);
  return input_args[kParametersIndex]->BuildType();
}
}  // namespace sgd

MIND_API_OPERATOR_IMPL(SGD, BaseOperator);
void SGD::Init(const float dampening, const float weight_decay, const bool nesterov) {
  set_nesterov(nesterov);
  set_dampening(dampening);
  set_weight_decay(weight_decay);
}

void SGD::set_dampening(const float dampening) {
  if (get_nesterov()) {
    (void)CheckAndConvertUtils::CheckValue<float>(kDampening, dampening, kEqual, 0.0, name());
  }
  (void)AddAttr(kDampening, api::MakeValue(dampening));
}

void SGD::set_weight_decay(const float weight_decay) { (void)AddAttr(kWeightDecay, api::MakeValue(weight_decay)); }

void SGD::set_nesterov(const bool nesterov) { (void)AddAttr(kNesterov, api::MakeValue(nesterov)); }

float SGD::get_dampening() const {
  auto value_ptr = GetAttr(kDampening);
  return GetValue<float>(value_ptr);
}

float SGD::get_weight_decay() const {
  auto value_ptr = GetAttr(kWeightDecay);
  return GetValue<float>(value_ptr);
}

bool SGD::get_nesterov() const {
  auto value_ptr = GetAttr(kNesterov);
  return GetValue<bool>(value_ptr);
}

abstract::AbstractBasePtr SGDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           SizeToLong(sgd::kSGDInputNum), op_name);
  auto types = sgd::SdgInferType(primitive, input_args);
  auto shapes = sgd::SgdInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGSGDInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return sgd::SgdInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return sgd::SdgInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SGDInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SGD, prim::kPrimSGD, AGSGDInfer, false);
}  // namespace ops
}  // namespace mindspore
