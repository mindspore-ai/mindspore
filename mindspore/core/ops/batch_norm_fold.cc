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

#include <vector>
#include <map>
#include <string>
#include "ops/batch_norm_fold.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void BatchNormFold::Init(const float momentum, const float epsilon, const bool is_training, const int64_t freeze_bn) {
  set_momentum(momentum);
  set_epsilon(epsilon);
  set_is_training(is_training);
  set_freeze_bn(freeze_bn);
}

void BatchNormFold::set_momentum(const float momentum) {
  CheckAndConvertUtils::CheckInRange<int64_t>(kMomentum, momentum, kIncludeBoth, {0.0, 1.0}, this->name());
  this->AddAttr(kMomentum, MakeValue(momentum));
}

float BatchNormFold::get_momentum() const {
  auto value_ptr = GetAttr(kMomentum);
  return GetValue<float>(value_ptr);
}

void BatchNormFold::set_epsilon(const float epsilon) {
  float match_value = 0.0;
  CheckAndConvertUtils::CheckValue(kEpsilon, epsilon, kGreaterThan, match_value, this->name());
  this->AddAttr(kEpsilon, MakeValue(epsilon));
}

float BatchNormFold::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

void BatchNormFold::set_is_training(const bool is_training) { this->AddAttr(kIsTraining, MakeValue(is_training)); }

bool BatchNormFold::get_is_training() const {
  auto value_ptr = GetAttr(kIsTraining);
  return GetValue<bool>(value_ptr);
}

void BatchNormFold::set_freeze_bn(const int64_t freeze_bn) { this->AddAttr(kFreezeBn, MakeValue(freeze_bn)); }

int64_t BatchNormFold::get_freeze_bn() const {
  auto value_ptr = GetAttr(kFreezeBn);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr BatchNormFoldInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto BatchNormFold_prim = primitive->cast<PrimBatchNormFoldPtr>();
  MS_EXCEPTION_IF_NULL(BatchNormFold_prim);
  auto op_name = BatchNormFold_prim->name();
  auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShape("mean_shape", input_args[1]->BuildShape(), op_name);
  auto variance_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("variance_shape", input_args[2]->BuildShape(), op_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), op_name);
  auto global_step_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("global_step_shape", input_args[3]->BuildShape(), op_name);
  CheckAndConvertUtils::Check("mean_shape", mean_shape, kEqual, "gamma_shape", variance_shape, op_name);
  CheckAndConvertUtils::Check("mean_shape[0]", mean_shape[0], kEqual, "input channel", x_shape[1], op_name);
  CheckAndConvertUtils::CheckInteger("global step shape len", global_step_shape.size(), kEqual, 1, op_name);

  auto mean_type = input_args[1]->BuildType();
  auto variance_type = input_args[2]->BuildType();
  auto x_type = input_args[0]->BuildType();
  auto global_step_type = input_args[3]->BuildType();

  std::map<std::string, TypePtr> args = {{"x", x_type}, {"mean", mean_type}, {"variance", variance_type}};
  CheckAndConvertUtils::CheckTensorTypeSame(args, {kNumberTypeFloat16, kNumberTypeFloat32}, op_name);
  CheckAndConvertUtils::CheckTensorTypeValid("gloabal_step", global_step_type, {kNumberTypeInt32}, op_name);

  auto tensor_type0 = x_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type0);
  auto element0 = tensor_type0->element();

  auto tensor_type1 = mean_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type1);
  auto element1 = tensor_type1->element();

  auto tensor_type2 = variance_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type2);
  auto element2 = tensor_type2->element();

  CheckAndConvertUtils::Check("input type", element0->type_id(), kEqual, "mean_type", element1->type_id(), op_name);
  CheckAndConvertUtils::Check("input type", element0->type_id(), kEqual, "variance_type", element2->type_id(), op_name);

  auto output = std::make_shared<abstract::AbstractTensor>(element0, mean_shape);
  AbstractBasePtrList output1 = {output, output, output, output};
  return std::make_shared<abstract::AbstractTuple>(output1);
}
REGISTER_PRIMITIVE_C(kNameBatchNormFold, BatchNormFold);
}  // namespace ops
}  // namespace mindspore
