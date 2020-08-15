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

#include "c_ops/bias_add.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> BiasAdd::GetAxis() const { return this->primitive->value.AsBiasAdd()->axis; }

void BiasAdd::SetAxis(const std::vector<int> &axis) { this->primitive->value.AsBiasAdd()->axis = axis; }

#else

std::vector<int> BiasAdd::GetAxis() const {
  auto fb_vector = this->primitive->value_as_BiasAdd()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void BiasAdd::SetAxis(const std::vector<int> &axis) {}
#endif
}  // namespace mindspore
