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

#include "c_ops/dropout.h"
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
void Dropout::Init(float keep_prob) { this->set_keep_prob(keep_prob); }
void Dropout::set_keep_prob(float keep_prob) {
  CheckAndConvertUtils::CheckInRange(kKeepProb, keep_prob, kIncludeRight, {0.0, 1.0}, this->name());
  this->AddAttr(kKeepProb, MakeValue(keep_prob));
}
float Dropout::get_keep_prob() {
  auto value_ptr = this->GetAttr(kKeepProb);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameDropout, Dropout);
}  // namespace mindspore
