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

#include "c_ops/upsample.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
string Upsample::GetMode() const { return this->primitive->value.AsUpsample()->mode; }
std::vector<float> Upsample::GetScales() const { return this->primitive->value.AsUpsample()->scales; }

void Upsample::SetMode(string mode) { this->primitive->value.AsUpsample()->mode = mode; }
void Upsample::SetScales(const std::vector<float> &scales) { this->primitive->value.AsUpsample()->scales = scales; }

#else

string Upsample::GetMode() const { return this->primitive->value_as_Upsample()->mode()->str(); }
std::vector<float> Upsample::GetScales() const {
  auto fb_vector = this->primitive->value_as_Upsample()->scales();
  return std::vector<float>(fb_vector->begin(), fb_vector->end());
}

void Upsample::SetMode(string mode) {}
void Upsample::SetScales(const std::vector<float> &scales) {}
#endif
}  // namespace mindspore
