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

#include "ops/custom_predict.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void CustomPredict::Init(const int64_t output_num, const float weight_threshold) {
  this->set_output_num(output_num);
  this->set_weight_threshold(weight_threshold);
}

void CustomPredict::set_output_num(const int64_t output_num) { this->AddAttr(kOutputNum, MakeValue(output_num)); }

int64_t CustomPredict::get_output_num() const {
  auto value_ptr = this->GetAttr(kOutputNum);
  return GetValue<int64_t>(value_ptr);
}

void CustomPredict::set_weight_threshold(const float weight_threshold) {
  this->AddAttr(kWeightThreshold, MakeValue(weight_threshold));
}

float CustomPredict::get_weight_threshold() const {
  auto value_ptr = this->GetAttr(kWeightThreshold);
  return GetValue<float>(value_ptr);
}

AbstractBasePtr CustomPredictInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto CustomPredict_prim = primitive->cast<PrimCustomPredictPtr>();
  MS_EXCEPTION_IF_NULL(CustomPredict_prim);
  for (auto input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  std::vector<int64_t> shape;
  shape.push_back(CustomPredict_prim->get_output_num());

  auto output0 = std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeInt32), shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeFloat32), shape);
  AbstractBasePtrList output = {output0, output1};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameCustomPredict, CustomPredict);
}  // namespace ops
}  // namespace mindspore
