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
#include "ops/resize_bilinear.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ResizeBilinear, BaseOperator);
void ResizeBilinear::set_size(const std::vector<int64_t> &size) { (void)this->AddAttr(kSize, api::MakeValue(size)); }

std::vector<int64_t> ResizeBilinear::get_size() const { return GetValue<std::vector<int64_t>>(GetAttr(kSize)); }

void ResizeBilinear::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

bool ResizeBilinear::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

bool ResizeBilinear::get_half_pixel_centers() const {
  auto value_ptr = GetAttr(kHalfPixelCenters);
  return GetValue<bool>(value_ptr);
}

void ResizeBilinear::Init(const std::vector<int64_t> &size, const bool align_corners) {
  this->set_size(size);
  this->set_align_corners(align_corners);
}

REGISTER_PRIMITIVE_C(kNameResizeBilinear, ResizeBilinear);
}  // namespace ops
}  // namespace mindspore
