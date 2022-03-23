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

REGISTER_PRIMITIVE_C(kNameBatchNorm, BatchNorm);
}  // namespace ops
}  // namespace mindspore
