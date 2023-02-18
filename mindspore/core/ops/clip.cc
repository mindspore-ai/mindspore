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

#include "ops/clip.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Clip, BaseOperator);
void Clip::Init(const float max, const float min) {
  this->set_max(max);
  this->set_min(min);
}

void Clip::set_max(const float max) { (void)this->AddAttr(kMax, api::MakeValue(max)); }

float Clip::get_max() const {
  auto value_ptr = this->GetAttr(kMax);
  return GetValue<float>(value_ptr);
}

void Clip::set_min(const float min) { (void)this->AddAttr(kMin, api::MakeValue(min)); }

float Clip::get_min() const {
  auto value_ptr = this->GetAttr(kMin);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameClip, Clip);
}  // namespace ops
}  // namespace mindspore
