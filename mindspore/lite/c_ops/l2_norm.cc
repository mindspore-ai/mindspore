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

#include "c_ops/l2_norm.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> L2Norm::GetAxis() const { return this->primitive->value.AsL2Norm()->axis; }
float L2Norm::GetEpsilon() const { return this->primitive->value.AsL2Norm()->epsilon; }

void L2Norm::SetAxis(const std::vector<int> &axis) { this->primitive->value.AsL2Norm()->axis = axis; }
void L2Norm::SetEpsilon(float epsilon) { this->primitive->value.AsL2Norm()->epsilon = epsilon; }

#else

std::vector<int> L2Norm::GetAxis() const {
  auto fb_vector = this->primitive->value_as_L2Norm()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
float L2Norm::GetEpsilon() const { return this->primitive->value_as_L2Norm()->epsilon(); }

void L2Norm::SetAxis(const std::vector<int> &axis) {}
void L2Norm::SetEpsilon(float epsilon) {}
#endif
}  // namespace mindspore
