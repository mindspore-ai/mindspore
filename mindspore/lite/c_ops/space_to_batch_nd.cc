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

#include "c_ops/space_to_batch_nd.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SpaceToBatchND::GetBlockShape() const { return this->primitive->value.AsSpaceToBatchND()->blockShape; }
std::vector<int> SpaceToBatchND::GetPaddings() const { return this->primitive->value.AsSpaceToBatchND()->paddings; }

void SpaceToBatchND::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive->value.AsSpaceToBatchND()->blockShape = block_shape;
}
void SpaceToBatchND::SetPaddings(const std::vector<int> &paddings) {
  this->primitive->value.AsSpaceToBatchND()->paddings = paddings;
}

#else

std::vector<int> SpaceToBatchND::GetBlockShape() const {
  auto fb_vector = this->primitive->value_as_SpaceToBatchND()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SpaceToBatchND::GetPaddings() const {
  auto fb_vector = this->primitive->value_as_SpaceToBatchND()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void SpaceToBatchND::SetBlockShape(const std::vector<int> &block_shape) {}
void SpaceToBatchND::SetPaddings(const std::vector<int> &paddings) {}
#endif
}  // namespace mindspore
