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
#include "ops/op_utils.h"
#include "ops/resize_nearest_neighbor.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void ResizeNearestNeighbor::Init(const std::vector<int64_t> &size, const bool align_corners) {
  this->set_size(size);
  this->set_align_corners(align_corners);
}
void ResizeNearestNeighbor::set_size(const std::vector<int64_t> &size) { this->AddAttr(kSize, MakeValue(size)); }
void ResizeNearestNeighbor::set_align_corners(const bool align_corners) {
  this->AddAttr(kAlignCorners, MakeValue(align_corners));
}
std::vector<int64_t> ResizeNearestNeighbor::get_size() const {
  auto value_ptr = GetAttr(kSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
bool ResizeNearestNeighbor::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameResizeNearestNeighbor, ResizeNearestNeighbor);
}  // namespace ops
}  // namespace mindspore
