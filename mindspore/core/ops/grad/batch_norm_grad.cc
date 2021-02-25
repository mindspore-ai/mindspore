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

#include <memory>
#include "ops/grad/batch_norm_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void BatchNormGrad::Init(const bool is_training, const float epsilon) {
  this->set_is_training(is_training);
  this->set_epsilon(epsilon);
}

void BatchNormGrad::set_epsilon(const float epsilon) {
  //  CheckAndConvertUtils::CheckInRange(kEpsilon, epsilon, kIncludeRight, {0, 1}, this->name());
  this->AddAttr(kEpsilon, MakeValue(epsilon));
}

float BatchNormGrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

void BatchNormGrad::set_is_training(const bool is_training) { this->AddAttr(kIsTraining, MakeValue(is_training)); }

bool BatchNormGrad::get_is_training() const {
  auto value_ptr = this->GetAttr(kIsTraining);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr BatchNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto BatchNormGrad_prim = primitive->cast<PrimBatchNormGradPtr>();
  MS_EXCEPTION_IF_NULL(BatchNormGrad_prim);
  auto op_name = BatchNormGrad_prim->name();
  MS_EXCEPTION_IF_NULL(input_args[1]);
  MS_EXCEPTION_IF_NULL(input_args[2]);
  MS_EXCEPTION_IF_NULL(input_args[3]);
  auto y_backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("y_backprop_shape", input_args[0]->BuildShape(), op_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[1]->BuildShape(), op_name);
  CheckAndConvertUtils::Check("BatchNorm y_backprop_shape", y_backprop_shape, kEqual, "BatchNorm x_shape", x_shape);

  auto dx = input_args[1]->Broaden();
  auto dscale = input_args[2]->Broaden();
  auto reserve_1 = input_args[3]->Broaden();
  auto reserve_2 = input_args[4]->Broaden();

  AbstractBasePtrList rets = {dx, dscale, dscale, reserve_1, reserve_2};
  return std::make_shared<abstract::AbstractTuple>(rets);
}
REGISTER_PRIMITIVE_C(kNameBatchNormGrad, BatchNormGrad);
}  // namespace ops
}  // namespace mindspore
