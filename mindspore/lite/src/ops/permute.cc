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

#include "src/ops/permute.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<long> Permute::GetOrder() const { return this->primitive->value.AsPermute()->order; }

void Permute::SetOrder(const std::vector<long> &order) { this->primitive->value.AsPermute()->order = order; }

#else

std::vector<long> Permute::GetOrder() const {
  auto fb_vector = this->primitive->value_as_Permute()->order();
  return std::vector<long>(fb_vector->begin(), fb_vector->end());
}

void Permute::SetOrder(const std::vector<long> &order) {}
#endif
}  // namespace lite
}  // namespace mindspore
