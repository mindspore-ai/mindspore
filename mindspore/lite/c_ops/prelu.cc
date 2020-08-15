/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "c_ops/prelu.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<float> Prelu::GetSlope() const { return this->primitive->value.AsPrelu()->slope; }

void Prelu::SetSlope(const std::vector<float> &slope) { this->primitive->value.AsPrelu()->slope = slope; }

#else

std::vector<float> Prelu::GetSlope() const {
  auto fb_vector = this->primitive->value_as_Prelu()->slope();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}

void Prelu::SetSlope(const std::vector<float> &slope) {}
#endif
}  // namespace mindspore
