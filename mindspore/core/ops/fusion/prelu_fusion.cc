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

#include "ops/fusion/prelu_fusion.h"
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void PReLUFusion::Init(const bool channel_shared, const std::vector<float> &slope) {
  this->set_channel_shared(channel_shared);
  this->set_slope(slope);
}

void PReLUFusion::set_channel_shared(const bool channel_shared) {
  this->AddAttr(kChannelShared, MakeValue(channel_shared));
}

void PReLUFusion::set_slope(const std::vector<float> &slope) { this->AddAttr(kSlope, MakeValue(slope)); }

bool PReLUFusion::get_channel_shared() const {
  auto value_ptr = GetAttr(kChannelShared);
  return GetValue<bool>(value_ptr);
}
std::vector<float> PReLUFusion::get_slope() const {
  auto value_ptr = GetAttr(kSlope);
  return GetValue<std::vector<float>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNamePReLUFusion, PReLUFusion);
}  // namespace ops
}  // namespace mindspore
