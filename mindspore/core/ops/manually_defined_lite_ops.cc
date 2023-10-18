/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/manually_defined_lite_ops.h"
#include <map>
#include <algorithm>
#include <memory>
#include <vector>
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
void BatchNorm::Init(const bool is_training, const float epsilon, const float momentum, const Format &format) {
  set_is_training(is_training);
  set_epsilon(epsilon);
  set_format(format);
  set_momentum(momentum);
}

void BatchNorm::set_is_training(const bool is_training) {
  (void)this->AddAttr(kIsTraining, api::MakeValue(is_training));
}

void BatchNorm::set_epsilon(const float epsilon) {
  CheckAndConvertUtils::CheckInRange<float>(kEpsilon, epsilon, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon));
}

void BatchNorm::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

void BatchNorm::set_momentum(const float momentun) {
  CheckAndConvertUtils::CheckInRange<float>(kMomentum, momentun, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kMomentum, api::MakeValue(momentun));
}

float BatchNorm::get_momentum() const {
  auto value_ptr = GetAttr(kMomentum);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

bool BatchNorm::get_is_training() const {
  auto value_ptr = GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

float BatchNorm::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

Format BatchNorm::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return Format(GetValue<int64_t>(value_ptr));
  }
  static const std::map<std::string, int64_t> valid_dataformat = {
    {"NHWC", Format::NHWC},
    {"NCHW", Format::NCHW},
  };
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = valid_dataformat.find(attr_value_str);
  if (iter == valid_dataformat.end()) {
    MS_LOG(EXCEPTION) << "for BatchNorm, Invalid format " << attr_value_str << ", use NHWC or NCHW";
  }
  return Format(iter->second);
}
MIND_API_OPERATOR_IMPL(BatchNorm, BaseOperator);
}  // namespace ops
}  // namespace mindspore
