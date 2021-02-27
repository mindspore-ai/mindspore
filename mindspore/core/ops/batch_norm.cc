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
#include <set>
#include <vector>
#include "ops/batch_norm.h"
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void BatchNorm::Init(const bool is_training, const float epsilon, const float momentum, const Format &format) {
  set_is_training(is_training);
  set_epsilon(epsilon);
  set_format(format);
  set_momentum(momentum);
}

void BatchNorm::set_is_training(const bool is_training) { this->AddAttr(kIsTraining, MakeValue(is_training)); }

void BatchNorm::set_epsilon(const float epsilon) {
  CheckAndConvertUtils::CheckInRange<float>(kEpsilon, epsilon, kIncludeBoth, {0.0, 1.0}, this->name());
  this->AddAttr(kEpsilon, MakeValue(epsilon));
}

void BatchNorm::set_format(const Format &format) {
  int64_t f = format;
  this->AddAttr(kFormat, MakeValue(f));
}

void BatchNorm::set_momentum(const float momentun) {
  CheckAndConvertUtils::CheckInRange<int64_t>(kMomentum, momentun, kIncludeBoth, {0.0, 1.0}, this->name());
  this->AddAttr(kMomentum, MakeValue(momentun));
}

float BatchNorm::get_momentum() const {
  auto value_ptr = GetAttr(kMomentum);
  return GetValue<float>(value_ptr);
}

bool BatchNorm::get_is_training() const {
  auto value_ptr = GetAttr(kIsTraining);
  return GetValue<bool>(value_ptr);
}

float BatchNorm::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

Format BatchNorm::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr BatchNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  // Infer shape
  MS_EXCEPTION_IF_NULL(primitive);
  auto batch_prim = primitive->cast<PrimBatchNormPtr>();
  MS_EXCEPTION_IF_NULL(batch_prim);
  auto prim_name = batch_prim->name();
  CheckAndConvertUtils::CheckInteger("batch_norm_infer", input_args.size(), kEqual, 5, prim_name);

  auto input_x = CheckAndConvertUtils::ConvertShapePtrToShape("input_x", input_args[0]->BuildShape(), prim_name);
  if (batch_prim->get_format() == NHWC) {
    input_x = {input_x[0], input_x[3], input_x[1], input_x[2]};
  }
  auto scale = CheckAndConvertUtils::ConvertShapePtrToShape("scale", input_args[1]->BuildShape(), prim_name);
  auto bias = CheckAndConvertUtils::ConvertShapePtrToShape("bias", input_args[2]->BuildShape(), prim_name);
  auto mean = CheckAndConvertUtils::ConvertShapePtrToShape("mean", input_args[3]->BuildShape(), prim_name);
  auto variance = CheckAndConvertUtils::ConvertShapePtrToShape("variance", input_args[4]->BuildShape(), prim_name);

  std::vector<int64_t> input_shape_norm;
  if (batch_prim->get_format() == NCHW) {
    input_shape_norm =
      CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->GetShapeTrack(), prim_name);
  } else {
    input_shape_norm.push_back(input_x[0]);
    input_shape_norm.push_back(input_x[3]);
    input_shape_norm.push_back(input_x[1]);
    input_shape_norm.push_back(input_x[2]);
  }
  CheckAndConvertUtils::CheckInteger("scale rank", scale.size(), kEqual, 1, prim_name);
  CheckAndConvertUtils::Check("scale shape", scale, kEqual, "bias shape", bias, prim_name, TypeError);
  CheckAndConvertUtils::Check("scale shape[0]", scale[0], kEqual, "input_x channel", input_shape_norm[1], prim_name,
                              TypeError);
  if (!batch_prim->get_is_training()) {
    CheckAndConvertUtils::CheckInteger("mean rank", mean.size(), kEqual, 1, prim_name);
    CheckAndConvertUtils::Check("mean shape", mean, kEqual, "variance shape", variance, prim_name, TypeError);
    CheckAndConvertUtils::Check("mean shape", mean, kEqual, "scale shape", scale, prim_name, TypeError);
  }

  // Infer type
  auto input_x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto scale_type = input_args[1]->BuildType()->cast<TensorTypePtr>()->element();
  auto bias_type = input_args[2]->BuildType()->cast<TensorTypePtr>()->element();

  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim_name);
  std::map<std::string, TypePtr> args;
  args.emplace("scale", input_args[1]->BuildType());
  args.emplace("bias", input_args[2]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  std::map<std::string, TypePtr> args_moving;
  args_moving.emplace("scale", input_args[2]->BuildType());
  args_moving.emplace("bias", input_args[3]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(args_moving, valid_types, prim_name);

  auto output0 = std::make_shared<abstract::AbstractTensor>(input_x_type, input_x);
  auto output1 = std::make_shared<abstract::AbstractTensor>(scale_type, scale);
  auto output2 = std::make_shared<abstract::AbstractTensor>(bias_type, scale);
  auto output3 = std::make_shared<abstract::AbstractTensor>(input_x_type, scale);
  if (batch_prim->get_format() == NHWC) {
    output2 = std::make_shared<abstract::AbstractTensor>(scale_type, scale);
    output3 = std::make_shared<abstract::AbstractTensor>(bias_type, scale);
    output1 = std::make_shared<abstract::AbstractTensor>(input_x_type, scale);
  }
  AbstractBasePtrList output = {output0, output1, output2, output3, output3};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameBatchNorm, BatchNorm);
}  // namespace ops
}  // namespace mindspore
