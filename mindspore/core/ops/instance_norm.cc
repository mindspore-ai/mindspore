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

#include "ops/instance_norm.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr InstanceNormInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  const auto gamma_shape_ptr = input_args[kInputIndex1]->BuildShape();
  const auto beta_shape_ptr = input_args[kInputIndex2]->BuildShape();
  const auto mean_shape_ptr = input_args[kInputIndex3]->BuildShape();
  const auto variance_shape_ptr = input_args[kInputIndex4]->BuildShape();

  if (input_x_shape_ptr->IsDynamic() || gamma_shape_ptr->IsDynamic() || beta_shape_ptr->IsDynamic() ||
      mean_shape_ptr->IsDynamic() || variance_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{input_x_shape_ptr, mean_shape_ptr, mean_shape_ptr});
  }

  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
  auto gamma_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(gamma_shape_ptr)[kShape];
  auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(beta_shape_ptr)[kShape];
  auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mean_shape_ptr)[kShape];
  auto variance_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(variance_shape_ptr)[kShape];

  constexpr size_t minimum_input_x_rank = 3;
  (void)CheckAndConvertUtils::CheckValue<size_t>("input_x rank", input_x_shape.size(), kGreaterEqual,
                                                 minimum_input_x_rank, prim_name);
  const size_t batch = LongToSize(input_x_shape[kInputIndex0]);
  const size_t channel = LongToSize(input_x_shape[kInputIndex1]);

  (void)CheckAndConvertUtils::CheckValue<size_t>("gamma rank", gamma_shape.size(), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("beta rank", beta_shape.size(), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("mean rank", mean_shape.size(), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("variance rank", variance_shape.size(), kEqual, 1, prim_name);

  (void)CheckAndConvertUtils::CheckValue<size_t>("gamma shape", LongToSize(gamma_shape[0]), kEqual, "(C, )", channel,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("beta shape", LongToSize(beta_shape[0]), kEqual, "(C, )", channel,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("mean shape", LongToSize(mean_shape[0]), kEqual, "(C, )", channel,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("variance shape", LongToSize(variance_shape[0]), kEqual, "(C, )",
                                                 channel, prim_name);

  const int64_t batch_channel = SizeToLong(batch * channel);
  abstract::ShapePtr save_mean_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{batch_channel});
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{input_x_shape_ptr, save_mean_shape, save_mean_shape});
}

TuplePtr InstanceNormInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto prim_name = primitive->name();
  const auto input_x = input_args[kInputIndex0]->BuildType();
  const auto gamma = input_args[kInputIndex1]->BuildType();
  const auto beta = input_args[kInputIndex2]->BuildType();
  const auto mean = input_args[kInputIndex3]->BuildType();
  const auto variance = input_args[kInputIndex4]->BuildType();

  (void)CheckAndConvertUtils::CheckTypeValid("input_x", input_x, {kFloat16, kFloat32}, prim_name);
  const std::map<std::string, TypePtr> types = {
    {"gamma", gamma},
    {"beta", beta},
    {"mean", mean},
    {"variance", variance},
  };
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_x, gamma, gamma});
}
}  // namespace
MIND_API_OPERATOR_IMPL(InstanceNorm, BaseOperator);
void InstanceNorm::Init(const float epsilon) { this->set_epsilon(epsilon); }

void InstanceNorm::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }
float InstanceNorm::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}
void InstanceNorm::set_momentum(const float momentum) { (void)this->AddAttr(kMomentum, api::MakeValue(momentum)); }
float InstanceNorm::get_momentum() const {
  auto value_ptr = GetAttr(kMomentum);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr InstanceNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto type = InstanceNormInferType(primitive, input_args);
  auto shape = InstanceNormInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(InstanceNorm, prim::kPrimInstanceNorm, InstanceNormInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
