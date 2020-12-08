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

#include "c_ops/custom_predict.h"
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
void CustomPredict::Init(int64_t outputNum, float weight_threshold) {
  this->set_outputNum(outputNum);
  this->set_weight_threshold(weight_threshold);
}

void CustomPredict::set_outputNum(int64_t outputNum) { this->AddAttr(kOutputNum, MakeValue(outputNum)); }

int64_t CustomPredict::get_outputNum() const {
  auto value_ptr = this->GetAttr(kOutputNum);
  return GetValue<int64_t>(value_ptr);
}

void CustomPredict::set_weight_threshold(float weight_threshold) {
  this->AddAttr(kWeightThreshold, MakeValue(weight_threshold));
}

float CustomPredict::get_weight_threshold() const {
  auto value_ptr = this->GetAttr(kWeightThreshold);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameCustomPredict, CustomPredict);
}  // namespace mindspore
