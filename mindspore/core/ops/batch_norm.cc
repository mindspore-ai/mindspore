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

void BatchNorm::set_is_training(const bool is_training) { (void)this->AddAttr(kIsTraining, MakeValue(is_training)); }

void BatchNorm::set_epsilon(const float epsilon) {
  CheckAndConvertUtils::CheckInRange<float>(kEpsilon, epsilon, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kEpsilon, MakeValue(epsilon));
}

void BatchNorm::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, MakeValue(f));
}

void BatchNorm::set_momentum(const float momentun) {
  CheckAndConvertUtils::CheckInRange<float>(kMomentum, momentun, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kMomentum, MakeValue(momentun));
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
  auto prim_name = primitive->name();
  const int64_t input_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  auto input_x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  if (format == NHWC) {
    input_x = {input_x[0], input_x[3], input_x[1], input_x[2]};
  }
  auto scale = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto bias = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto mean = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto variance = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];

  std::vector<int64_t> input_shape_norm;
  if (format == NCHW) {
    input_shape_norm = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  } else {
    input_shape_norm.push_back(input_x[0]);
    input_shape_norm.push_back(input_x[3]);
    input_shape_norm.push_back(input_x[1]);
    input_shape_norm.push_back(input_x[2]);
  }
  (void)CheckAndConvertUtils::CheckInteger("scale rank", SizeToLong(scale.size()), kEqual, 1, prim_name);
  CheckAndConvertUtils::Check("scale shape", scale, kEqual, "bias shape", bias, prim_name, TypeError);
  CheckAndConvertUtils::Check("scale shape[0]", scale[0], kEqual, "input_x channel", input_shape_norm[1], prim_name,
                              TypeError);

  if (!GetValue<bool>(primitive->GetAttr(kIsTraining))) {
    (void)CheckAndConvertUtils::CheckInteger("mean rank", SizeToLong(mean.size()), kEqual, 1, prim_name);
    CheckAndConvertUtils::Check("mean shape", mean, kEqual, "variance shape", variance, prim_name, TypeError);
    CheckAndConvertUtils::Check("mean shape", mean, kEqual, "scale shape", scale, prim_name, TypeError);
  }

  // Infer type
  auto scale_type = input_args[kInputIndex1]->BuildType()->cast<TensorTypePtr>()->element();
  auto bias_type = input_args[kInputIndex2]->BuildType()->cast<TensorTypePtr>()->element();

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto input_x_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), valid_types, prim_name);
  std::map<std::string, TypePtr> args;
  (void)args.emplace("scale", input_args[kInputIndex1]->BuildType());
  (void)args.emplace("bias", input_args[kInputIndex2]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  std::map<std::string, TypePtr> args_moving;
  (void)args_moving.emplace("scale", input_args[kInputIndex2]->BuildType());
  (void)args_moving.emplace("bias", input_args[kInputIndex3]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args_moving, valid_types, prim_name);

  auto output0 = std::make_shared<abstract::AbstractTensor>(input_x_type, input_x);
  auto output1 = std::make_shared<abstract::AbstractTensor>(scale_type, scale);
  auto output2 = std::make_shared<abstract::AbstractTensor>(bias_type, scale);
  auto output3 = std::make_shared<abstract::AbstractTensor>(input_x_type, scale);
  if (format == NHWC) {
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
